from typing import List, Dict, Any
import sqlite3
import os

DB_FILE = "products.db"
class ProductsDB:
    def __init__(self, db_path: str):
        #self.db_path = db_path
        self.db_path = DB_FILE
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database with products table and version tracking."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS db_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Check current version
            cursor.execute("SELECT MAX(version) FROM db_version")
            current_version = cursor.fetchone()[0] or 0

            # Apply version 1 schema if not already applied
            if current_version < 1:
                cursor.execute("""
                    CREATE TABLE products (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        url TEXT,
                        keywords TEXT,
                        type TEXT,
                        description TEXT,
                        content TEXT
                    )
                """)
                cursor.execute("INSERT INTO db_version (version) VALUES (1)")
                conn.commit()

            # Add type column if not exists
            cursor.execute("PRAGMA table_info(products)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'type' not in columns:
                cursor.execute("ALTER TABLE products ADD COLUMN type TEXT")
                conn.commit()

    def add_product(self, title: str, url: str, keywords: str, type: str, description: str, content: str) -> int:
        """Add a new product to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO products (title, url, keywords, type, description, content)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (title, url, keywords, type, description, content))
            conn.commit()
            return cursor.lastrowid

    def update_product(self, product_id: int, title: str, url: str, keywords: str,
                      type: str, description: str, content: str):
        """Update an existing product."""
        product_id = int(product_id)
        print(f"Debug - Updating product ID: {product_id}")
        print(f"Debug - Values: title={title}, url={url}, keywords={keywords}, type={type}, description={description}, content={content}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM products")
            all_ids = [row[0] for row in cursor.fetchall()]
            print(f"Debug - All product IDs in database: {all_ids}")
            # Vérifier si l'ID existe
            cursor.execute("SELECT id FROM products WHERE id = ?", (product_id,))
            if not cursor.fetchone():
                print(f"Debug - Product ID {product_id} not found in database")
                raise ValueError(f"Product with ID {product_id} does not exist")
            # Exécuter la mise à jour
            cursor.execute("""
                UPDATE products
                SET title = ?, url = ?, keywords = ?, type = ?, description = ?, content = ?
                WHERE id = ?
            """, (title, url, keywords, type, description, content, product_id))
            print(f"Debug - Rows affected: {cursor.rowcount}")
            conn.commit()
            if cursor.rowcount == 0:
                print(f"Debug - No rows updated for ID: {product_id}")

    def delete_product(self, product_id: int):
        """Delete a product from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM products WHERE id = ?", (int(product_id),))
            conn.commit()

    def get_product(self, product_id: int) -> Dict[str, Any]:
        """Get a single product by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM products WHERE id = ?", (int(product_id),))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "title": row[1],
                    "url": row[2],
                    "keywords": row[3],
                    "type": row[4],
                    "description": row[5],
                    "content": row[6]
                }
            return None

    def get_all_products(self) -> List[Dict[str, Any]]:
        """Get all products from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM products")
            rows = cursor.fetchall()
            return [{
                "id": row[0],
                "title": row[1],
                "url": row[2],
                "keywords": row[3],
                "type": row[4],
                "description": row[5],
                "content": row[6]
            } for row in rows]

    def update_product_field(self, product_id: int, field: str, value: str):
        """Update a single field for an existing product."""
        product_id = int(product_id)  # Forcer la conversion en entier
        print(f"Debug - Updating field {field} for product ID: {product_id}")
        print(f"Debug - New value: {value}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Vérifier si l'ID existe
            cursor.execute("SELECT id FROM products WHERE id = ?", (product_id,))
            if not cursor.fetchone():
                print(f"Debug - Product ID {product_id} not found in database")
                raise ValueError(f"Product with ID {product_id} does not exist")
            # Vérifier que le champ est valide
            valid_fields = ['title', 'url', 'keywords', 'type', 'description', 'content']
            if field not in valid_fields:
                raise ValueError(f"Invalid field: {field}")
            # Exécuter la mise à jour
            cursor.execute(f"""
                UPDATE products
                SET {field} = ?
                WHERE id = ?
            """, (value, product_id))
            print(f"Debug - Rows affected: {cursor.rowcount}")
            conn.commit()
            if cursor.rowcount == 0:
                print(f"Debug - No rows updated for ID: {product_id}")
