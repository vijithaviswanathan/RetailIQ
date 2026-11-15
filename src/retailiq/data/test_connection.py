from sqlalchemy import create_engine, text

# Connect to the local PostgreSQL database
engine = create_engine("postgresql+psycopg:///retailiq")

try:
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT
                    current_database(),
                    current_user,
                    version();
            """)
        )

        for row in result:
            print("Database:", row[0])
            print("User:", row[1])
            print("PostgreSQL:", row[2])

        schemas = connection.execute(
            text("""
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name IN ('hillstrom', 'retail')
                ORDER BY schema_name;
            """)
        )

        print("\nAvailable schemas:")
        for schema in schemas:
            print("-", schema[0])

    print("\nPostgreSQL connection successful!")

except Exception as error:
    print("Connection failed:")
    print(error)