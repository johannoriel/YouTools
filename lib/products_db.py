from typing import List, Dict, Any
import sqlite3
import os

DB_FILE = "products.db"
class ProductsDB:
    def __init__(self):
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
                        source TEXT,
                        goal TEXT,
                        related TEXT,
                        description TEXT,
                        content TEXT
                    )
                """)
                cursor.execute("INSERT INTO db_version (version) VALUES (1)")
                conn.commit()

    def add_product(self, title: str, url: str, keywords: str, type: str, source: str, goal: str, related: str, description: str, content: str) -> int:
        """Add a new product to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO products (title, url, keywords, type, source, goal, related, description, content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (title, url, keywords, type, source, goal, related, description, content))
            conn.commit()
            return cursor.lastrowid

    def update_product(self, product_id: int, title: str, url: str, keywords: str,
                      type: str, source: str, goal: str, related: str, description: str, content: str):
        """Update an existing product."""
        product_id = int(product_id)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM products WHERE id = ?", (product_id,))
            if not cursor.fetchone():
                raise ValueError(f"Product with ID {product_id} does not exist")
            cursor.execute("""
                UPDATE products
                SET title = ?, url = ?, keywords = ?, type = ?, source = ?, goal = ?, related = ?, description = ?, content = ?
                WHERE id = ?
            """, (title, url, keywords, type, source, goal, related, description, content, product_id))
            conn.commit()

    def update_or_add_product(self, product_id: int | None, title: str, url: str, keywords: str,
                             type: str, source: str, goal: str, related: str, description: str, content: str) -> int:
        """Update an existing product or add a new one if product_id is None."""
        if product_id is not None:
            self.update_product(product_id, title, url, keywords, type, source, goal, related, description, content)
            return product_id
        return self.add_product(title, url, keywords, type, source, goal, related, description, content)

    def delete_product(self, product_id: int):
        """Delete a product from the database."""
        product_id = int(product_id)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM products WHERE id = ?", (product_id,))
            conn.commit()

    def get_product(self, product_id: int) -> Dict[str, Any]:
        """Get a single product by ID."""
        product_id = int(product_id)
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
                    "type": row[4],
                    "source": row[5],
                    "goal": row[6],
                    "related": row[7],
                    "description": row[8],
                    "content": row[9]
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
                "source": row[5],
                "goal": row[6],
                "related": row[7],
                "description": row[8],
                "content": row[9]
            } for row in rows]

    def update_product_field(self, product_id: int, field: str, value: str):
        """Update a single field for an existing product."""
        product_id = int(product_id)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM products WHERE id = ?", (product_id,))
            if not cursor.fetchone():
                raise ValueError(f"Product with ID {product_id} does not exist")
            valid_fields = ['title', 'url', 'keywords', 'type', 'source', 'goal', 'related', 'description', 'content']
            if field not in valid_fields:
                raise ValueError(f"Invalid field: {field}")
            cursor.execute(f"""
                UPDATE products
                SET {field} = ?
                WHERE id = ?
            """, (value, product_id))
            conn.commit()
