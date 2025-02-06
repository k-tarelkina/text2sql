import json
import os

class DatabaseCatalog:
    def __init__(self, file_path=os.path.join('.', 'data', 'tables.json')):
        with open(file_path, 'r') as file:
            self.tables = json.load(file)

        for database in self.tables:
            database['schema'] = self._get_database_schema(database)


    def _get_database_schema(self, database):
        sql = ""

        for table_id, table in enumerate(database['table_names_original']):
            sql += f"\nCREATE TABLE {table} ("

            for column_id, column_data in enumerate(zip(database['column_names_original'], database['column_types'])):
                column_table_id_name, column_type = column_data
                column_table_id, column_name = column_table_id_name

                if column_table_id == table_id:
                    sql += f"\n\t{column_name} {column_type}"

                    if column_id in database['primary_keys']:
                        sql += ' primary key'

                    for id, reference_id in database['foreign_keys']:
                        if column_id == id:
                            reference_column = database['column_names_original'][reference_id]
                            reference_table = database["table_names_original"][reference_column[0]]
                            sql += f' foreign key ({column_name}) references {reference_table}({reference_column[1]})'

            sql+="\n)"
        return sql.strip()
    
    def get_database_schema_by_id(self, database_id):
        for db in self.tables:
            if db['db_id'] == database_id:
                return db['schema']
        return None
    
DATABASE_CATALOG = DatabaseCatalog()