import sqlite3
from datetime import datetime


class BenchDB:
    def __init__(self, db_path="bench_results.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialise la base de données avec la table des résultats et une contrainte d'unicité."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bench_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    server_url TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT,  -- Maintenant raw_response
                    expected TEXT,
                    score INTEGER DEFAULT 0 CHECK (score >= 0 AND score <= 5),
                    execution_time REAL,
                    raw_response_length INTEGER,
                    converted_response_length INTEGER,
                    shortened_response_length INTEGER,
                    timestamp TEXT,
                    UNIQUE(model, server_url, prompt)  -- Contrainte d'unicité
                )
            """)
            conn.commit()

    def save_result(self, model, server_url, prompt, response, expected, execution_time, raw_len, conv_len, short_len):
        """Sauvegarde ou met à jour un résultat dans la base de données si valide."""
        if not response or "LLM Error" in response:
            print(
                f"DEBUG: Not saving result for {model} at {server_url} - prompt: {prompt} - invalid response: {response}")
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Si existe déjà, on met à jour au lieu d'insérer
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO bench_results (model, server_url, prompt, response, expected, execution_time,
                                                         raw_response_length, converted_response_length, shortened_response_length, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (model, server_url, prompt, response, expected, execution_time, raw_len, conv_len, short_len, datetime.now().isoformat()))
                conn.commit()
                print(
                    f"DEBUG: Saved/Updated result for {model} at {server_url} - prompt: {prompt} - ID: {cursor.lastrowid}")
                return True
        except sqlite3.Error as e:
            print(
                f"DEBUG: Database error while saving result for {model} at {server_url} - prompt: {prompt} - Error: {str(e)}")
            return False

    def update_score(self, result_id, score):
        """Met à jour le score d'un résultat."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE bench_results SET score = ? WHERE id = ?", (score, result_id))
                conn.commit()
                print(
                    f"DEBUG: Updated score for result ID {result_id} to {score}")
        except sqlite3.Error as e:
            print(
                f"DEBUG: Database error while updating score for ID {result_id} - Error: {str(e)}")

    def get_result(self, model, server_url, prompt):
        """Récupère un résultat spécifique depuis la base."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, response, expected, score, execution_time, raw_response_length, converted_response_length, shortened_response_length
                    FROM bench_results
                    WHERE model = ? AND server_url = ? AND prompt = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (model, server_url, prompt))
                result = cursor.fetchone()
                if result:
                    return {
                        "id": result[0], "response": result[1], "expected": result[2], "score": result[3],
                        "execution_time": result[4], "raw_response_length": result[5],
                        "converted_response_length": result[6], "shortened_response_length": result[7]
                    }
                print(
                    f"DEBUG: No result found for {model} at {server_url} - prompt: {prompt}")
                return None
        except sqlite3.Error as e:
            print(
                f"DEBUG: Database error while fetching result for {model} at {server_url} - prompt: {prompt} - Error: {str(e)}")
            return None

    def get_results_by_model(self, model, server_url):
        """Récupère tous les résultats pour un modèle et un serveur donné."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT prompt, response, expected, score, execution_time, raw_response_length, converted_response_length, shortened_response_length
                    FROM bench_results
                    WHERE model = ? AND server_url = ?
                    ORDER BY timestamp DESC
                """, (model, server_url))
                results = [{"prompt": row[0], "response": row[1], "expected": row[2], "score": row[3],
                           "execution_time": row[4], "raw_response_length": row[5],
                            "converted_response_length": row[6], "shortened_response_length": row[7]}
                           for row in cursor.fetchall()]
                print(
                    f"DEBUG: Fetched {len(results)} results for {model} at {server_url}")
                return results
        except sqlite3.Error as e:
            print(
                f"DEBUG: Database error while fetching results for {model} at {server_url} - Error: {str(e)}")
            return []

    def get_total_score(self, model, server_url):
        """Calcule le score total pour un modèle et un serveur."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT SUM(score)
                    FROM bench_results
                    WHERE model = ? AND server_url = ?
                """, (model, server_url))
                total = cursor.fetchone()[0]
                return total if total is not None else 0
        except sqlite3.Error as e:
            print(
                f"DEBUG: Database error while calculating total score for {model} at {server_url} - Error: {str(e)}")
            return 0

    def get_all_models(self):
        """Récupère tous les modèles uniques avec leur serveur."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT DISTINCT model, server_url FROM bench_results")
                return [(row[0], row[1]) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(
                f"DEBUG: Database error while fetching all models - Error: {str(e)}")
            return []
