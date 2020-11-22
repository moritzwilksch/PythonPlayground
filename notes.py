import sqlite3

class ScrapyToSQLPipeline:
    def __init__(self):
        self.connection = sqlite3.connect("ISDatabase.db")
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS isdata
            (
                id INTEGER PRIMARY KEY, 
                stadtteil VARCHAR(128),
                name VARCHAR(128),
                link VARCHAR(512),
                privatehh INTEGER,
                commercials INTEGER,
                residentsage VARCHAR(10),
                durationofresidence VARCHAR(10),
                families VARCHAR(10),
                singles VARCHAR(10)
            )
            """
        )

    def process_item(self, item, spider):
        pass

ScrapyToSQLPipeline()