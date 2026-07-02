import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from typing import List, Dict, Optional
import os
import secrets
import threading
import pickle
import bcrypt
import json
from datetime import datetime

class DatabaseManager:
    """Postgres access layer with per-thread connections.

    A single libpq connection/cursor is NOT safe to share across threads.
    This object is used from many threads at once (the frame-processing
    worker threads plus FastAPI's request threads), so each thread gets its
    own lazily-created connection and cursor, exposed through the ``conn``
    and ``cursor`` properties. Existing call sites that use ``self.conn`` /
    ``self.cursor`` keep working unchanged — they simply resolve to the
    calling thread's connection.
    """

    def __init__(self, host=None, database=None, user=None, password=None):
        # Fall back to environment variables (then local-dev defaults)
        self.connection_params = {
            'host': host or os.environ.get("DB_HOST", "localhost"),
            'database': database or os.environ.get("DB_NAME", "facial_recognition"),
            'user': user or os.environ.get("DB_USER", "postgres"),
            'password': password or os.environ.get("DB_PASSWORD", ""),
            # Disable libpq TLS for localhost. Postgres 17 negotiates TLS 1.3 by
            # default; psycopg2-binary's bundled libpq/OpenSSL collides with the
            # OpenSSL that torch/tensorflow/faiss have already loaded in-process
            # and segfaults during connect. Local sockets don't need TLS.
            'sslmode': 'disable',
        }
        # Per-thread connection/cursor storage.
        self._local = threading.local()

    def _ensure_connection(self):
        """Create (or recreate) this thread's connection/cursor if needed."""
        conn = getattr(self._local, 'conn', None)
        if conn is None or conn.closed:
            conn = psycopg2.connect(**self.connection_params)
            self._local.conn = conn
            self._local.cursor = conn.cursor(cursor_factory=RealDictCursor)
        return self._local.conn

    @property
    def conn(self):
        return self._ensure_connection()

    @property
    def cursor(self):
        self._ensure_connection()
        return self._local.cursor

    def connect(self):
        """Establish this thread's database connection and ensure schema exists."""
        try:
            self._ensure_connection()
            print("Database connection established")
            # Create tables if they don't exist
            self._create_tables()
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def disconnect(self):
        """Close the current thread's database connection."""
        cursor = getattr(self._local, 'cursor', None)
        conn = getattr(self._local, 'conn', None)
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        self._local.cursor = None
        self._local.conn = None

    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            # Create persons table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    employee_id VARCHAR(50) UNIQUE NOT NULL,
                    department VARCHAR(100),
                    authorized_zones TEXT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create face embeddings table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                    embedding BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create access logs table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_logs (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER REFERENCES persons(id) ON DELETE CASCADE,
                    camera_id VARCHAR(50),
                    confidence FLOAT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create license plates table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS license_plates (
                    id SERIAL PRIMARY KEY,
                    plate_number VARCHAR(20) UNIQUE NOT NULL,
                    vehicle_type VARCHAR(50),
                    owner_name VARCHAR(255),
                    owner_id VARCHAR(50),
                    is_authorized BOOLEAN DEFAULT TRUE,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expiry_date TIMESTAMP,
                    notes TEXT
                )
            """)
            
            # Create vehicle access logs table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_access_logs (
                    id SERIAL PRIMARY KEY,
                    plate_number VARCHAR(20),
                    camera_id VARCHAR(50),
                    confidence FLOAT,
                    vehicle_type VARCHAR(50),
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT,
                    is_authorized BOOLEAN,
                    FOREIGN KEY (plate_number) REFERENCES license_plates(plate_number) ON DELETE CASCADE
                )
            """)
            
            # Create users table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(20) NOT NULL DEFAULT 'user',
                    full_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Explicitly revoked JWT tokens (logout / forced invalidation).
            # Enables instant revocation of a session before its natural expiry
            # (exp). Expired entries can be purged periodically.
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    jti        UUID PRIMARY KEY,
                    user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TIMESTAMP NOT NULL,
                    revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires
                ON revoked_tokens (expires_at)
            """)

            # Create alarms table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS alarms (
                    id SERIAL PRIMARY KEY,
                    camera_id VARCHAR(50) NOT NULL,
                    type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
                    status VARCHAR(20) NOT NULL DEFAULT 'unresolved',
                    description TEXT,
                    snapshot TEXT,
                    detection_metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    resolved_by VARCHAR(255),
                    notes TEXT
                )
            """)

            # Create zones table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS zones (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    is_restricted BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create cameras table (persistent camera registry)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id SERIAL PRIMARY KEY,
                    camera_id VARCHAR(50) UNIQUE NOT NULL,
                    name VARCHAR(255),
                    zone_id INTEGER REFERENCES zones(id) ON DELETE SET NULL,
                    location VARCHAR(255),
                    stream_url TEXT,
                    camera_type VARCHAR(20) DEFAULT 'ip',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create detection logs table (historical record of every detection)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_logs (
                    id SERIAL PRIMARY KEY,
                    camera_id VARCHAR(50) NOT NULL,
                    type VARCHAR(20) NOT NULL,
                    subject VARCHAR(255),
                    confidence FLOAT,
                    severity VARCHAR(20),
                    status VARCHAR(30),
                    details JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate_number ON license_plates(plate_number)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_detlogs_created_at ON detection_logs(created_at DESC)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_detlogs_type ON detection_logs(type)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_detlogs_camera ON detection_logs(camera_id)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_access_time ON vehicle_access_logs(detected_at)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_camera ON vehicle_access_logs(camera_id)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_alarms_status ON alarms(status)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_alarms_type ON alarms(type)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_alarms_severity ON alarms(severity)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_alarms_created_at ON alarms(created_at)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_cameras_zone ON cameras(zone_id)")

            self.conn.commit()
            print("Database tables created/verified")

            # Seed default admin user if no users exist
            self._seed_default_admin()
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating tables: {e}")
            raise
            
    # Original person-related methods
    def add_person(self, name: str, employee_id: str, department: str = None, 
                   authorized_zones: List[str] = None) -> int:
        """Add a new person to the database"""
        try:
            query = """
                INSERT INTO persons (name, employee_id, department, authorized_zones)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """
            self.cursor.execute(query, (name, employee_id, department, authorized_zones))
            self.conn.commit()
            return self.cursor.fetchone()['id']
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding person: {e}")
            raise
            
    def add_face_embedding(self, person_id: int, embedding: np.ndarray):
        """Store face embedding for a person"""
        try:
            # Serialize the numpy array
            embedding_bytes = pickle.dumps(embedding)
            
            query = """
                INSERT INTO face_embeddings (person_id, embedding)
                VALUES (%s, %s)
            """
            self.cursor.execute(query, (person_id, psycopg2.Binary(embedding_bytes)))
            self.conn.commit()
            print(f"Embedding added for person_id: {person_id}")
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding face embedding: {e}")
            raise
            
    def get_all_embeddings(self) -> List[Dict]:
        """Retrieve all face embeddings with person information"""
        try:
            query = """
                SELECT fe.id, fe.person_id, fe.embedding, p.name, p.employee_id
                FROM face_embeddings fe
                JOIN persons p ON fe.person_id = p.id
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            # Deserialize embeddings
            for result in results:
                result['embedding'] = pickle.loads(result['embedding'])
                
            return results
        except Exception as e:
            print(f"Error retrieving embeddings: {e}")
            raise
            
    def log_access(self, person_id: int, camera_id: str, confidence: float):
        """Log an access event"""
        try:
            # Convert numpy float to Python float
            confidence = float(confidence)
            
            query = """
                INSERT INTO access_logs (person_id, camera_id, confidence)
                VALUES (%s, %s, %s)
            """
            self.cursor.execute(query, (person_id, camera_id, confidence))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error logging access: {e}")
            raise
            
    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """Get person information by ID"""
        try:
            query = "SELECT * FROM persons WHERE id = %s"
            self.cursor.execute(query, (person_id,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error retrieving person: {e}")
            raise
            
    # ── Person management methods ─────────────────────────────────

    def list_persons(self, search: str = None, department: str = None,
                     limit: int = 50, offset: int = 0) -> Dict:
        """List persons with optional search/filter and pagination"""
        try:
            conditions = []
            params = []

            if search:
                conditions.append("(LOWER(p.name) LIKE LOWER(%s) OR LOWER(p.employee_id) LIKE LOWER(%s))")
                search_pattern = f"%{search}%"
                params.extend([search_pattern, search_pattern])

            if department:
                conditions.append("LOWER(p.department) = LOWER(%s)")
                params.append(department)

            where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""

            # Get total count
            count_query = f"SELECT COUNT(*) as count FROM persons p{where_clause}"
            self.cursor.execute(count_query, params)
            total = self.cursor.fetchone()['count']

            # Get paginated results with face count and last seen
            query = f"""
                SELECT p.*,
                       COALESCE(fe.face_count, 0) as face_count,
                       al.last_seen
                FROM persons p
                LEFT JOIN (
                    SELECT person_id, COUNT(*) as face_count
                    FROM face_embeddings
                    GROUP BY person_id
                ) fe ON p.id = fe.person_id
                LEFT JOIN (
                    SELECT person_id, MAX(detected_at) as last_seen
                    FROM access_logs
                    GROUP BY person_id
                ) al ON p.id = al.person_id
                {where_clause}
                ORDER BY p.created_at DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            self.cursor.execute(query, params)
            persons = self.cursor.fetchall()

            return {"persons": persons, "total": total}
        except Exception as e:
            print(f"Error listing persons: {e}")
            raise

    def update_person(self, person_id: int, name: str = None,
                      department: str = None,
                      authorized_zones: List[str] = None) -> bool:
        """Update person details"""
        try:
            updates = []
            params = []
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if department is not None:
                updates.append("department = %s")
                params.append(department)
            if authorized_zones is not None:
                updates.append("authorized_zones = %s")
                params.append(authorized_zones)
            if not updates:
                return False
            params.append(person_id)
            query = f"UPDATE persons SET {', '.join(updates)} WHERE id = %s"
            self.cursor.execute(query, params)
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating person: {e}")
            raise

    def delete_person(self, person_id: int) -> bool:
        """Delete a person (cascades to embeddings and access logs)"""
        try:
            query = "DELETE FROM persons WHERE id = %s"
            self.cursor.execute(query, (person_id,))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting person: {e}")
            raise

    def count_person_embeddings(self, person_id: int) -> int:
        """Count face embeddings for a person"""
        try:
            query = "SELECT COUNT(*) as count FROM face_embeddings WHERE person_id = %s"
            self.cursor.execute(query, (person_id,))
            return self.cursor.fetchone()['count']
        except Exception as e:
            print(f"Error counting embeddings: {e}")
            raise

    def get_person_access_history(self, person_id: int, limit: int = 20) -> List[Dict]:
        """Get recent access history for a person"""
        try:
            query = """
                SELECT camera_id, confidence, detected_at
                FROM access_logs
                WHERE person_id = %s
                ORDER BY detected_at DESC
                LIMIT %s
            """
            self.cursor.execute(query, (person_id, limit))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error getting access history: {e}")
            raise

    def get_all_departments(self) -> List[str]:
        """Get all unique departments"""
        try:
            query = "SELECT DISTINCT department FROM persons WHERE department IS NOT NULL AND department != '' ORDER BY department"
            self.cursor.execute(query)
            return [row['department'] for row in self.cursor.fetchall()]
        except Exception as e:
            print(f"Error getting departments: {e}")
            raise

    # ── License plate management methods ────────────────────────

    def list_license_plates(self, search: str = None, vehicle_type: str = None,
                            is_authorized: bool = None, limit: int = 50,
                            offset: int = 0) -> Dict:
        """List license plates with optional search/filter and pagination"""
        try:
            conditions = []
            params = []

            if search:
                conditions.append(
                    "(LOWER(lp.plate_number) LIKE LOWER(%s) OR LOWER(lp.owner_name) LIKE LOWER(%s))"
                )
                search_pattern = f"%{search}%"
                params.extend([search_pattern, search_pattern])

            if vehicle_type:
                conditions.append("LOWER(lp.vehicle_type) = LOWER(%s)")
                params.append(vehicle_type)

            if is_authorized is not None:
                conditions.append("lp.is_authorized = %s")
                params.append(is_authorized)

            where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""

            # Get total count
            count_query = f"SELECT COUNT(*) as count FROM license_plates lp{where_clause}"
            self.cursor.execute(count_query, params)
            total = self.cursor.fetchone()['count']

            # Get paginated results with last detection info
            query = f"""
                SELECT lp.*,
                       val.last_seen,
                       COALESCE(val.detection_count, 0) as detection_count
                FROM license_plates lp
                LEFT JOIN (
                    SELECT plate_number,
                           MAX(detected_at) as last_seen,
                           COUNT(*) as detection_count
                    FROM vehicle_access_logs
                    GROUP BY plate_number
                ) val ON lp.plate_number = val.plate_number
                {where_clause}
                ORDER BY lp.registration_date DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            self.cursor.execute(query, params)
            plates = self.cursor.fetchall()

            return {"plates": plates, "total": total}
        except Exception as e:
            print(f"Error listing license plates: {e}")
            raise

    def update_license_plate(self, plate_number: str, owner_name: str = None,
                             owner_id: str = None, vehicle_type: str = None,
                             is_authorized: bool = None, expiry_date=None,
                             notes: str = None) -> bool:
        """Update license plate details"""
        try:
            updates = []
            params = []
            if owner_name is not None:
                updates.append("owner_name = %s")
                params.append(owner_name)
            if owner_id is not None:
                updates.append("owner_id = %s")
                params.append(owner_id)
            if vehicle_type is not None:
                updates.append("vehicle_type = %s")
                params.append(vehicle_type)
            if is_authorized is not None:
                updates.append("is_authorized = %s")
                params.append(is_authorized)
            if expiry_date is not None:
                updates.append("expiry_date = %s")
                params.append(expiry_date if expiry_date != '' else None)
            if notes is not None:
                updates.append("notes = %s")
                params.append(notes)
            if not updates:
                return False
            params.append(plate_number)
            query = f"UPDATE license_plates SET {', '.join(updates)} WHERE plate_number = %s"
            self.cursor.execute(query, params)
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating license plate: {e}")
            raise

    def delete_license_plate(self, plate_number: str) -> bool:
        """Delete a license plate (cascades to access logs via FK)"""
        try:
            query = "DELETE FROM license_plates WHERE plate_number = %s"
            self.cursor.execute(query, (plate_number,))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting license plate: {e}")
            raise

    def get_plate_access_history(self, plate_number: str, limit: int = 20) -> List[Dict]:
        """Get recent access history for a license plate"""
        try:
            query = """
                SELECT camera_id, confidence, vehicle_type, detected_at, is_authorized
                FROM vehicle_access_logs
                WHERE plate_number = %s
                ORDER BY detected_at DESC
                LIMIT %s
            """
            self.cursor.execute(query, (plate_number, limit))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error getting plate access history: {e}")
            raise

    def get_all_vehicle_types(self) -> List[str]:
        """Get all unique vehicle types"""
        try:
            query = "SELECT DISTINCT vehicle_type FROM license_plates WHERE vehicle_type IS NOT NULL AND vehicle_type != '' ORDER BY vehicle_type"
            self.cursor.execute(query)
            return [row['vehicle_type'] for row in self.cursor.fetchall()]
        except Exception as e:
            print(f"Error getting vehicle types: {e}")
            raise

    # License plate related methods
    def add_license_plate(self, plate_number: str, vehicle_type: str = None,
                         owner_name: str = None, owner_id: str = None,
                         is_authorized: bool = True, expiry_date: datetime = None,
                         notes: str = None) -> int:
        """Add a new license plate to the database"""
        try:
            query = """
                INSERT INTO license_plates 
                (plate_number, vehicle_type, owner_name, owner_id, is_authorized, expiry_date, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            self.cursor.execute(query, (plate_number, vehicle_type, owner_name, 
                                      owner_id, is_authorized, expiry_date, notes))
            self.conn.commit()
            return self.cursor.fetchone()['id']
        except psycopg2.IntegrityError:
            self.conn.rollback()
            print(f"License plate {plate_number} already exists")
            raise
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding license plate: {e}")
            raise
            
    def get_license_plate(self, plate_number: str) -> Optional[Dict]:
        """Get license plate information"""
        try:
            query = "SELECT * FROM license_plates WHERE plate_number = %s"
            self.cursor.execute(query, (plate_number,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error retrieving license plate: {e}")
            raise

    def lookup_owner_by_plate(self, plate_number: str) -> Optional[str]:
        """Get owner name by plate number"""
        try:
            plate_info = self.get_license_plate(plate_number)
            return plate_info['owner_name'] if plate_info else None
        except Exception as e:
            print(f"Error looking up owner for plate {plate_number}: {e}")
            return None
            
    def update_license_plate_authorization(self, plate_number: str, is_authorized: bool):
        """Update license plate authorization status"""
        try:
            query = """
                UPDATE license_plates 
                SET is_authorized = %s 
                WHERE plate_number = %s
            """
            self.cursor.execute(query, (is_authorized, plate_number))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating license plate authorization: {e}")
            raise
            
    def log_vehicle_access(self, plate_number: str, camera_id: str, 
                          confidence: float, vehicle_type: str = None,
                          image_path: str = None):
        """Log a vehicle access event"""
        try:
            # Check if plate exists and get authorization status
            plate_info = self.get_license_plate(plate_number)
            
            if plate_info:
                is_authorized = plate_info['is_authorized']
            else:
                # Unknown plate
                is_authorized = False
                # Optionally add unknown plate to database
                self.add_license_plate(plate_number, vehicle_type, is_authorized=False,
                                     notes="Auto-detected unknown plate")
                
            query = """
                INSERT INTO vehicle_access_logs 
                (plate_number, camera_id, confidence, vehicle_type, image_path, is_authorized)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(query, (plate_number, camera_id, float(confidence), 
                                      vehicle_type, image_path, is_authorized))
            self.conn.commit()
            
            return is_authorized
            
        except Exception as e:
            self.conn.rollback()
            print(f"Error logging vehicle access: {e}")
            raise
            
    def get_vehicle_access_logs(self, limit: int = 100, 
                               camera_id: str = None,
                               start_time: datetime = None,
                               end_time: datetime = None,
                               authorized_only: bool = None) -> List[Dict]:
        """Get vehicle access logs with filters"""
        try:
            query = """
                SELECT val.*, lp.owner_name, lp.owner_id 
                FROM vehicle_access_logs val
                LEFT JOIN license_plates lp ON val.plate_number = lp.plate_number
                WHERE 1=1
            """
            params = []
            
            if camera_id:
                query += " AND val.camera_id = %s"
                params.append(camera_id)
                
            if start_time:
                query += " AND val.detected_at >= %s"
                params.append(start_time)
                
            if end_time:
                query += " AND val.detected_at <= %s"
                params.append(end_time)
                
            if authorized_only is not None:
                query += " AND val.is_authorized = %s"
                params.append(authorized_only)
                
            query += " ORDER BY val.detected_at DESC LIMIT %s"
            params.append(limit)
            
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
            
        except Exception as e:
            print(f"Error retrieving vehicle access logs: {e}")
            raise
            
    def get_all_authorized_plates(self) -> List[str]:
        """Get all authorized license plates"""
        try:
            query = """
                SELECT plate_number 
                FROM license_plates 
                WHERE is_authorized = TRUE 
                AND (expiry_date IS NULL OR expiry_date > CURRENT_TIMESTAMP)
            """
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return [r['plate_number'] for r in results]
        except Exception as e:
            print(f"Error retrieving authorized plates: {e}")
            raise
            
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {}
            
            # Person statistics
            self.cursor.execute("SELECT COUNT(*) as count FROM persons")
            stats['total_persons'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT COUNT(*) as count FROM face_embeddings")
            stats['total_face_embeddings'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT COUNT(*) as count FROM access_logs")
            stats['total_face_accesses'] = self.cursor.fetchone()['count']
            
            # Vehicle statistics
            self.cursor.execute("SELECT COUNT(*) as count FROM license_plates")
            stats['total_plates'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT COUNT(*) as count FROM license_plates WHERE is_authorized = TRUE")
            stats['authorized_plates'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("SELECT COUNT(*) as count FROM vehicle_access_logs")
            stats['total_vehicle_accesses'] = self.cursor.fetchone()['count']
            
            self.cursor.execute("""
                SELECT COUNT(*) as count
                FROM vehicle_access_logs
                WHERE is_authorized = FALSE
            """)
            stats['unauthorized_vehicle_accesses'] = self.cursor.fetchone()['count']

            # Alarm statistics
            self.cursor.execute("SELECT COUNT(*) as count FROM alarms WHERE status = 'unresolved'")
            stats['unresolved_alarms'] = self.cursor.fetchone()['count']

            self.cursor.execute("SELECT COUNT(*) as count FROM alarms WHERE status = 'unresolved' AND severity = 'critical'")
            stats['critical_alarms'] = self.cursor.fetchone()['count']

            self.cursor.execute("SELECT COUNT(*) as count FROM alarms")
            stats['total_alarms'] = self.cursor.fetchone()['count']

            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            raise

    # ── Alarm management methods ───────────────────────────────────

    def create_alarm(self, camera_id: str, alarm_type: str, severity: str,
                     description: str = None, snapshot: str = None,
                     detection_metadata: Dict = None) -> int:
        """Create a new alarm"""
        try:
            query = """
                INSERT INTO alarms (camera_id, type, severity, description, snapshot, detection_metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            meta_json = json.dumps(detection_metadata) if detection_metadata else None
            self.cursor.execute(query, (camera_id, alarm_type, severity, description,
                                        snapshot, meta_json))
            self.conn.commit()
            return self.cursor.fetchone()['id']
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating alarm: {e}")
            raise

    def get_recent_alarm(self, camera_id: str, alarm_type: str,
                         cooldown_seconds: int = 30,
                         dedup_subject: str = None) -> Optional[Dict]:
        """Get the most recent matching alarm within the cooldown window (for
        deduplication). When `dedup_subject` is given, only alarms whose stored
        detection_metadata->>'dedup_subject' matches are considered, so distinct
        concurrent subjects (two intruders, knife vs gun) aren't collapsed."""
        try:
            # NOTE: the interval must be built with `(%s || ' seconds')::interval`
            # — a `%s` placeholder inside a quoted literal (INTERVAL '%s seconds')
            # is NOT parameterized by psycopg2 and silently breaks.
            params = [camera_id, alarm_type, str(cooldown_seconds)]
            subject_clause = ""
            if dedup_subject is not None:
                subject_clause = " AND detection_metadata->>'dedup_subject' = %s"
                params.append(dedup_subject)
            query = f"""
                SELECT * FROM alarms
                WHERE camera_id = %s AND type = %s AND status = 'unresolved'
                  AND created_at > NOW() - (%s || ' seconds')::interval
                  {subject_clause}
                ORDER BY created_at DESC
                LIMIT 1
            """
            self.cursor.execute(query, params)
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error getting recent alarm: {e}")
            raise

    def list_alarms(self, status: str = None, alarm_type: str = None,
                    severity: str = None, camera_id: str = None,
                    limit: int = 50, offset: int = 0) -> Dict:
        """List alarms with filters and pagination"""
        try:
            conditions = []
            params = []

            if status:
                conditions.append("status = %s")
                params.append(status)
            if alarm_type:
                conditions.append("type = %s")
                params.append(alarm_type)
            if severity:
                conditions.append("severity = %s")
                params.append(severity)
            if camera_id:
                conditions.append("camera_id = %s")
                params.append(camera_id)

            where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""

            count_query = f"SELECT COUNT(*) as count FROM alarms{where_clause}"
            self.cursor.execute(count_query, params)
            total = self.cursor.fetchone()['count']

            query = f"""
                SELECT id, camera_id, type, severity, status, description,
                       detection_metadata, created_at, resolved_at, resolved_by, notes
                FROM alarms
                {where_clause}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            self.cursor.execute(query, params)
            alarms = self.cursor.fetchall()

            return {"alarms": alarms, "total": total}
        except Exception as e:
            print(f"Error listing alarms: {e}")
            raise

    def get_alarm(self, alarm_id: int) -> Optional[Dict]:
        """Get a single alarm by ID (including snapshot)"""
        try:
            query = "SELECT * FROM alarms WHERE id = %s"
            self.cursor.execute(query, (alarm_id,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error getting alarm: {e}")
            raise

    def update_alarm(self, alarm_id: int, status: str = None,
                     notes: str = None, resolved_by: str = None) -> bool:
        """Update alarm status and/or notes"""
        try:
            updates = []
            params = []
            if status is not None:
                updates.append("status = %s")
                params.append(status)
                if status in ('resolved', 'false_alarm'):
                    updates.append("resolved_at = NOW()")
                    if resolved_by:
                        updates.append("resolved_by = %s")
                        params.append(resolved_by)
            if notes is not None:
                updates.append("notes = %s")
                params.append(notes)
            if not updates:
                return False
            params.append(alarm_id)
            query = f"UPDATE alarms SET {', '.join(updates)} WHERE id = %s"
            self.cursor.execute(query, params)
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating alarm: {e}")
            raise

    def get_alarm_stats(self) -> Dict:
        """Get alarm statistics for dashboard"""
        try:
            stats = {}
            self.cursor.execute("SELECT COUNT(*) as count FROM alarms WHERE status = 'unresolved'")
            stats['unresolved'] = self.cursor.fetchone()['count']

            self.cursor.execute("SELECT COUNT(*) as count FROM alarms WHERE status = 'unresolved' AND severity = 'critical'")
            stats['critical_unresolved'] = self.cursor.fetchone()['count']

            self.cursor.execute("SELECT COUNT(*) as count FROM alarms WHERE status = 'resolved'")
            stats['resolved'] = self.cursor.fetchone()['count']

            self.cursor.execute("SELECT COUNT(*) as count FROM alarms WHERE status = 'false_alarm'")
            stats['false_alarm'] = self.cursor.fetchone()['count']

            self.cursor.execute("""
                SELECT type, COUNT(*) as count FROM alarms
                WHERE status = 'unresolved'
                GROUP BY type
            """)
            stats['by_type'] = {row['type']: row['count'] for row in self.cursor.fetchall()}

            return stats
        except Exception as e:
            print(f"Error getting alarm stats: {e}")
            raise

    def bulk_update_alarms(self, alarm_ids: List[int], status: str,
                           resolved_by: str = None) -> int:
        """Bulk update alarm statuses"""
        try:
            if status in ('resolved', 'false_alarm'):
                if resolved_by:
                    query = """
                        UPDATE alarms SET status = %s, resolved_at = NOW(), resolved_by = %s
                        WHERE id = ANY(%s)
                    """
                    self.cursor.execute(query, (status, resolved_by, alarm_ids))
                else:
                    query = """
                        UPDATE alarms SET status = %s, resolved_at = NOW()
                        WHERE id = ANY(%s)
                    """
                    self.cursor.execute(query, (status, alarm_ids))
            else:
                query = "UPDATE alarms SET status = %s WHERE id = ANY(%s)"
                self.cursor.execute(query, (status, alarm_ids))
            self.conn.commit()
            return self.cursor.rowcount
        except Exception as e:
            self.conn.rollback()
            print(f"Error bulk updating alarms: {e}")
            raise

    # ── Detection log methods ────────────────────────────────────

    def insert_detection_log(self, camera_id: str, log_type: str,
                             subject: str = None, confidence: float = None,
                             severity: str = None, status: str = None,
                             details: Dict = None) -> int:
        """Persist a single detection event for historical browsing."""
        try:
            query = """
                INSERT INTO detection_logs
                    (camera_id, type, subject, confidence, severity, status, details)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """
            conf = float(confidence) if confidence is not None else None
            details_json = json.dumps(details) if details else None
            self.cursor.execute(query, (camera_id, log_type, subject, conf,
                                        severity, status, details_json))
            row = self.cursor.fetchone()
            self.conn.commit()
            return row['id']
        except Exception as e:
            self.conn.rollback()
            print(f"Error inserting detection log: {e}")
            raise

    def list_detection_logs(self, log_type: str = None, camera_id: str = None,
                            status: str = None, search: str = None,
                            date_from: str = None, date_to: str = None,
                            limit: int = 50, offset: int = 0) -> Dict:
        """List detection logs with filters + pagination."""
        try:
            conditions = []
            params = []

            if log_type:
                conditions.append("type = %s")
                params.append(log_type)
            if camera_id:
                conditions.append("LOWER(camera_id) LIKE LOWER(%s)")
                params.append(f"%{camera_id}%")
            if status:
                conditions.append("status = %s")
                params.append(status)
            if search:
                conditions.append("LOWER(subject) LIKE LOWER(%s)")
                params.append(f"%{search}%")
            if date_from:
                conditions.append("created_at >= %s")
                params.append(date_from)
            if date_to:
                conditions.append("created_at <= %s")
                params.append(date_to)

            where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""

            count_query = f"SELECT COUNT(*) as count FROM detection_logs{where_clause}"
            self.cursor.execute(count_query, params)
            total = self.cursor.fetchone()['count']

            query = f"""
                SELECT id, camera_id, type, subject, confidence,
                       severity, status, details, created_at
                FROM detection_logs
                {where_clause}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])
            self.cursor.execute(query, params)
            logs = self.cursor.fetchall()

            return {"logs": logs, "total": total}
        except Exception as e:
            print(f"Error listing detection logs: {e}")
            raise

    def get_detection_log_stats(self, hours: int = 24) -> Dict:
        """Per-type counts over the last N hours + total."""
        try:
            stats = {}
            self.cursor.execute(
                "SELECT COUNT(*) as count FROM detection_logs "
                "WHERE created_at > NOW() - (%s || ' hours')::interval",
                (str(hours),)
            )
            stats['total_recent'] = self.cursor.fetchone()['count']

            self.cursor.execute(
                "SELECT type, COUNT(*) as count FROM detection_logs "
                "WHERE created_at > NOW() - (%s || ' hours')::interval "
                "GROUP BY type",
                (str(hours),)
            )
            stats['by_type'] = {row['type']: row['count'] for row in self.cursor.fetchall()}

            self.cursor.execute("SELECT COUNT(*) as count FROM detection_logs")
            stats['total'] = self.cursor.fetchone()['count']
            stats['window_hours'] = hours
            return stats
        except Exception as e:
            print(f"Error getting detection log stats: {e}")
            raise

    def get_detection_log_timeseries(self, hours: int = 24) -> Dict:
        """Time-bucketed detection counts per type for charting.

        Uses hourly buckets for hours <= 48, else daily buckets.
        Returns empty buckets too so the chart has no gaps.
        """
        try:
            bucket = 'hour' if hours <= 48 else 'day'
            step = '1 hour' if bucket == 'hour' else '1 day'
            self.cursor.execute(
                f"""
                WITH buckets AS (
                    SELECT generate_series(
                        date_trunc('{bucket}', NOW() - (%s || ' hours')::interval),
                        date_trunc('{bucket}', NOW()),
                        %s::interval
                    ) AS ts
                ),
                counts AS (
                    SELECT date_trunc('{bucket}', created_at) AS ts, type, COUNT(*) AS c
                    FROM detection_logs
                    WHERE created_at > NOW() - (%s || ' hours')::interval
                    GROUP BY 1, 2
                )
                SELECT b.ts AS ts, c.type AS type, COALESCE(c.c, 0) AS count
                FROM buckets b
                LEFT JOIN counts c ON c.ts = b.ts
                ORDER BY b.ts
                """,
                (str(hours), step, str(hours))
            )
            rows = self.cursor.fetchall()
            buckets_set = {}
            types = set()
            for row in rows:
                ts = row['ts'].isoformat() if row['ts'] else None
                if ts is None:
                    continue
                if ts not in buckets_set:
                    buckets_set[ts] = {}
                if row['type']:
                    buckets_set[ts][row['type']] = row['count']
                    types.add(row['type'])
            series = [
                {'ts': ts, 'total': sum(counts.values()), 'by_type': counts}
                for ts, counts in sorted(buckets_set.items())
            ]
            return {'bucket': bucket, 'hours': hours, 'series': series,
                    'types': sorted(types)}
        except Exception as e:
            print(f"Error getting detection log timeseries: {e}")
            raise

    def get_detection_log_breakdown(self, hours: int = 24) -> Dict:
        """Per-camera and per-type breakdowns over the last N hours."""
        try:
            self.cursor.execute(
                "SELECT camera_id, COUNT(*) AS count FROM detection_logs "
                "WHERE created_at > NOW() - (%s || ' hours')::interval "
                "GROUP BY camera_id ORDER BY count DESC",
                (str(hours),)
            )
            by_camera = [{'camera_id': r['camera_id'] or 'unknown', 'count': r['count']}
                         for r in self.cursor.fetchall()]

            self.cursor.execute(
                "SELECT type, COUNT(*) AS count FROM detection_logs "
                "WHERE created_at > NOW() - (%s || ' hours')::interval "
                "GROUP BY type ORDER BY count DESC",
                (str(hours),)
            )
            by_type = [{'type': r['type'], 'count': r['count']}
                       for r in self.cursor.fetchall()]

            self.cursor.execute(
                "SELECT severity, COUNT(*) AS count FROM detection_logs "
                "WHERE created_at > NOW() - (%s || ' hours')::interval "
                "AND severity IS NOT NULL "
                "GROUP BY severity ORDER BY count DESC",
                (str(hours),)
            )
            by_severity = [{'severity': r['severity'], 'count': r['count']}
                           for r in self.cursor.fetchall()]

            return {'hours': hours, 'by_camera': by_camera,
                    'by_type': by_type, 'by_severity': by_severity}
        except Exception as e:
            print(f"Error getting detection log breakdown: {e}")
            raise

    # ── Zones management methods ────────────────────────────────

    def create_zone(self, name: str, description: str = None,
                    is_restricted: bool = False) -> int:
        try:
            self.cursor.execute(
                """INSERT INTO zones (name, description, is_restricted)
                   VALUES (%s, %s, %s) RETURNING id""",
                (name, description, is_restricted)
            )
            self.conn.commit()
            return self.cursor.fetchone()['id']
        except psycopg2.IntegrityError:
            self.conn.rollback()
            raise ValueError(f"Zone '{name}' already exists")
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating zone: {e}")
            raise

    def list_zones(self) -> List[Dict]:
        try:
            self.cursor.execute("""
                SELECT z.id, z.name, z.description, z.is_restricted, z.created_at,
                       COALESCE(c.camera_count, 0) AS camera_count
                FROM zones z
                LEFT JOIN (
                    SELECT zone_id, COUNT(*) AS camera_count
                    FROM cameras WHERE zone_id IS NOT NULL
                    GROUP BY zone_id
                ) c ON c.zone_id = z.id
                ORDER BY z.name
            """)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error listing zones: {e}")
            raise

    def get_zone(self, zone_id: int) -> Optional[Dict]:
        try:
            self.cursor.execute("SELECT * FROM zones WHERE id = %s", (zone_id,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error getting zone: {e}")
            raise

    def get_zone_by_name(self, name: str) -> Optional[Dict]:
        try:
            self.cursor.execute("SELECT * FROM zones WHERE name = %s", (name,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error getting zone by name: {e}")
            raise

    def update_zone(self, zone_id: int, name: str = None,
                    description: str = None, is_restricted: bool = None) -> bool:
        try:
            old = self.get_zone(zone_id)
            if not old:
                return False
            updates = []
            params = []
            if name is not None and name != old['name']:
                updates.append("name = %s")
                params.append(name)
            if description is not None:
                updates.append("description = %s")
                params.append(description)
            if is_restricted is not None:
                updates.append("is_restricted = %s")
                params.append(is_restricted)
            if not updates:
                return False
            params.append(zone_id)
            self.cursor.execute(
                f"UPDATE zones SET {', '.join(updates)} WHERE id = %s",
                params
            )
            # If renamed, update persons.authorized_zones arrays to keep names in sync
            if name is not None and name != old['name']:
                self.cursor.execute(
                    """UPDATE persons
                       SET authorized_zones = array_replace(authorized_zones, %s, %s)
                       WHERE %s = ANY(authorized_zones)""",
                    (old['name'], name, old['name'])
                )
            self.conn.commit()
            return True
        except psycopg2.IntegrityError:
            self.conn.rollback()
            raise ValueError(f"Zone name '{name}' already exists")
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating zone: {e}")
            raise

    def delete_zone(self, zone_id: int) -> bool:
        try:
            zone = self.get_zone(zone_id)
            if not zone:
                return False
            self.cursor.execute("DELETE FROM zones WHERE id = %s", (zone_id,))
            # Remove zone name from any persons.authorized_zones arrays
            self.cursor.execute(
                """UPDATE persons
                   SET authorized_zones = array_remove(authorized_zones, %s)
                   WHERE %s = ANY(authorized_zones)""",
                (zone['name'], zone['name'])
            )
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting zone: {e}")
            raise

    # ── Cameras registry methods ────────────────────────────────

    def upsert_camera(self, camera_id: str, name: str = None,
                      zone_id: int = None, location: str = None,
                      stream_url: str = None, camera_type: str = 'ip') -> int:
        """Insert or update a camera registry entry. Only updates non-None fields on conflict."""
        try:
            self.cursor.execute(
                """INSERT INTO cameras (camera_id, name, zone_id, location, stream_url, camera_type)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (camera_id) DO UPDATE SET
                     name = COALESCE(EXCLUDED.name, cameras.name),
                     zone_id = COALESCE(EXCLUDED.zone_id, cameras.zone_id),
                     location = COALESCE(EXCLUDED.location, cameras.location),
                     stream_url = COALESCE(EXCLUDED.stream_url, cameras.stream_url),
                     camera_type = COALESCE(EXCLUDED.camera_type, cameras.camera_type)
                   RETURNING id""",
                (camera_id, name, zone_id, location, stream_url, camera_type)
            )
            self.conn.commit()
            return self.cursor.fetchone()['id']
        except Exception as e:
            self.conn.rollback()
            print(f"Error upserting camera: {e}")
            raise

    def list_cameras_db(self) -> List[Dict]:
        try:
            self.cursor.execute("""
                SELECT c.id, c.camera_id, c.name, c.zone_id, c.location,
                       c.stream_url, c.camera_type, c.created_at,
                       z.name AS zone_name, z.is_restricted AS zone_is_restricted
                FROM cameras c
                LEFT JOIN zones z ON z.id = c.zone_id
                ORDER BY c.camera_id
            """)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error listing cameras: {e}")
            raise

    def get_camera(self, camera_id: str) -> Optional[Dict]:
        try:
            self.cursor.execute("""
                SELECT c.*, z.name AS zone_name, z.is_restricted AS zone_is_restricted
                FROM cameras c
                LEFT JOIN zones z ON z.id = c.zone_id
                WHERE c.camera_id = %s
            """, (camera_id,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error getting camera: {e}")
            raise

    def update_camera(self, camera_id: str, name: str = None,
                      zone_id=..., location: str = None,
                      stream_url: str = None) -> bool:
        """Update fields on a camera. Pass zone_id=None to clear."""
        try:
            updates = []
            params = []
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if zone_id is not ...:
                updates.append("zone_id = %s")
                params.append(zone_id)
            if location is not None:
                updates.append("location = %s")
                params.append(location)
            if stream_url is not None:
                updates.append("stream_url = %s")
                params.append(stream_url)
            if not updates:
                return False
            params.append(camera_id)
            self.cursor.execute(
                f"UPDATE cameras SET {', '.join(updates)} WHERE camera_id = %s",
                params
            )
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating camera: {e}")
            raise

    def delete_camera(self, camera_id: str) -> bool:
        try:
            self.cursor.execute("DELETE FROM cameras WHERE camera_id = %s", (camera_id,))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting camera: {e}")
            raise

    def get_camera_zone(self, camera_id: str) -> Optional[Dict]:
        """Return {zone_id, zone_name, is_restricted} for a camera, or None if unassigned."""
        try:
            self.cursor.execute("""
                SELECT z.id AS zone_id, z.name AS zone_name, z.is_restricted
                FROM cameras c
                JOIN zones z ON z.id = c.zone_id
                WHERE c.camera_id = %s
            """, (camera_id,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error getting camera zone: {e}")
            raise

    def get_person_by_employee_id(self, employee_id: str) -> Optional[Dict]:
        try:
            self.cursor.execute(
                "SELECT * FROM persons WHERE employee_id = %s",
                (employee_id,)
            )
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error getting person by employee_id: {e}")
            raise

    # ── User management methods ──────────────────────────────────

    def _seed_default_admin(self):
        """Create the default admin user if no users exist.

        The password is taken from the ADMIN_PASSWORD environment variable. If it
        is not set, a random one is generated and printed once at startup, so no
        password is ever hardcoded in source.
        """
        try:
            self.cursor.execute("SELECT COUNT(*) as count FROM users")
            count = self.cursor.fetchone()['count']
            if count == 0:
                password = os.environ.get("ADMIN_PASSWORD")
                generated = password is None
                if generated:
                    password = secrets.token_urlsafe(12)
                self.create_user('admin', password, 'admin', 'Administrator')
                if generated:
                    print("Default admin user 'admin' created with a generated "
                          f"password: {password}")
                    print("Set ADMIN_PASSWORD in the environment to choose your own, "
                          "and change this password after first login.")
                else:
                    print("Default admin user 'admin' created (password from ADMIN_PASSWORD).")
        except Exception as e:
            self.conn.rollback()
            print(f"Error seeding default admin: {e}")

    @staticmethod
    def _hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def _verify_password(password: str, password_hash: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

    def create_user(self, username: str, password: str, role: str = 'user',
                    full_name: str = None) -> int:
        """Create a new user"""
        try:
            password_hash = self._hash_password(password)
            query = """
                INSERT INTO users (username, password_hash, role, full_name)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """
            self.cursor.execute(query, (username, password_hash, role, full_name))
            self.conn.commit()
            return self.cursor.fetchone()['id']
        except psycopg2.IntegrityError:
            self.conn.rollback()
            raise ValueError(f"Username '{username}' already exists")
        except Exception as e:
            self.conn.rollback()
            print(f"Error creating user: {e}")
            raise

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user info (without password_hash)"""
        try:
            query = "SELECT * FROM users WHERE username = %s"
            self.cursor.execute(query, (username,))
            user = self.cursor.fetchone()
            if user and self._verify_password(password, user['password_hash']):
                return {
                    'id': user['id'],
                    'username': user['username'],
                    'role': user['role'],
                    'full_name': user['full_name'],
                    'created_at': user['created_at']
                }
            return None
        except Exception as e:
            print(f"Error authenticating user: {e}")
            raise

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID (without password_hash)"""
        try:
            query = "SELECT id, username, role, full_name, created_at FROM users WHERE id = %s"
            self.cursor.execute(query, (user_id,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error getting user: {e}")
            raise

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username (without password_hash)"""
        try:
            query = "SELECT id, username, role, full_name, created_at FROM users WHERE username = %s"
            self.cursor.execute(query, (username,))
            return self.cursor.fetchone()
        except Exception as e:
            print(f"Error getting user: {e}")
            raise

    def list_users(self) -> List[Dict]:
        """List all users (without password_hash)"""
        try:
            query = "SELECT id, username, role, full_name, created_at FROM users ORDER BY created_at"
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error listing users: {e}")
            raise

    def update_user(self, user_id: int, role: str = None, full_name: str = None,
                    password: str = None) -> bool:
        """Update user details"""
        try:
            updates = []
            params = []
            if role is not None:
                updates.append("role = %s")
                params.append(role)
            if full_name is not None:
                updates.append("full_name = %s")
                params.append(full_name)
            if password is not None:
                updates.append("password_hash = %s")
                params.append(self._hash_password(password))
            if not updates:
                return False
            params.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = %s"
            self.cursor.execute(query, params)
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating user: {e}")
            raise

    def delete_user(self, user_id: int) -> bool:
        """Delete a user"""
        try:
            query = "DELETE FROM users WHERE id = %s"
            self.cursor.execute(query, (user_id,))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting user: {e}")
            raise

    def revoke_token(self, jti: str, user_id: int, expires_at) -> None:
        """Add a token (by jti) to the revocation denylist."""
        try:
            self.cursor.execute(
                """
                INSERT INTO revoked_tokens (jti, user_id, expires_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (jti) DO NOTHING
                """,
                (jti, user_id, expires_at),
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error revoking token: {e}")
            raise

    def is_token_revoked(self, jti: str) -> bool:
        """Return True if the jti has been revoked (logout / forced invalidation)."""
        if not jti:
            return False
        self.cursor.execute(
            "SELECT 1 FROM revoked_tokens WHERE jti = %s", (jti,)
        )
        return self.cursor.fetchone() is not None

    def purge_expired_revocations(self) -> int:
        """Delete already-expired entries (no longer in effect). Returns row count."""
        self.cursor.execute(
            "DELETE FROM revoked_tokens WHERE expires_at < NOW()"
        )
        deleted = self.cursor.rowcount
        self.conn.commit()
        return deleted