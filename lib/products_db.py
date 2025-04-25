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
                        description TEXT,
                        content TEXT
                    )
                """)
                cursor.execute("INSERT INTO db_version (version) VALUES (1)")
                conn.commit()

    def add_product(self, title: str, url: str, keywords: str, description: str, content: str) -> int:
        """Add a new product to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO products (title, url, keywords, description, content)
                VALUES (?, ?, ?, ?, ?)
            """, (title, url, keywords, description, content))
            conn.commit()
            return cursor.lastrowid

    # lib/products_db.py

    def update_product(self, product_id: int, title: str, url: str, keywords: str,
                        description: str, content: str):
        """Update an existing product."""
        print(f"Debug - Updating product ID: {product_id}")
        print(f"Debug - Values: title={title}, url={url}, keywords={keywords}, description={description}, content={content}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM products")
            all_ids = [row[0] for row in cursor.fetchall()]
            print(f"Debug - All product IDs in database: {all_ids}")
            # Vérifier si l'ID existe
            cursor.execute("SELECT id FROM products WHERE id = ?", (int(product_id),))
            if not cursor.fetchone():
                print(f"Debug - Product ID {product_id} not found in database")
                raise ValueError(f"Product with ID {product_id} does not exist")
            # Exécuter la mise à jour
            cursor.execute("""
                UPDATE products
                SET title = ?, url = ?, keywords = ?, description = ?, content = ?
                WHERE id = ?
            """, (title, url, keywords, description, content, int(product_id)))
            print(f"Debug - Rows affected: {cursor.rowcount}")
            conn.commit()
            if cursor.rowcount == 0:
                print(f"Debug - No rows updated for ID: {product_id}")

    def delete_product(self, product_id: int):
        """Delete a product from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM products WHERE id = ?", (product_id,))
            conn.commit()

    def get_product(self, product_id: int) -> Dict[str, Any]:
        """Get a single product by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "title": row[1],
                    "url": row[2],
                    "keywords": row[3],
                    "description": row[4],
                    "content": row[5]
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
                "description": row[4],
                "content": row[5]
            } for row in rows]
