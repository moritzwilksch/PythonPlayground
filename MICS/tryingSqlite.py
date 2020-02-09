# %%
import sqlite3

# %%
con = sqlite3.connect("sqlite-lokal.db")
curser = con.cursor()

# %%
def insert_all_new_into(table, rows):
    curser.execute("select * from product where price < 7")
    res = curser.fetchall()
    pks = [x[0] for x in res]
    for r in rows:
        try:
            curser.execute(f"""
            INSERT INTO {table} 
            VALUES {r}
            """)
        except sqlite3.IntegrityError:
            continue
    con.commit()


products = [
    (1, 'Breakfast small', 4.9, 1),
    (2, 'Breakfast medium', 6.9, 1),
    (4, 'Coffee', 2.9, 2),
    (5, 'Latte Macchiato', 3.9, 2)
]
insert_all_new_into("Product", products)

categories = [
    (1, 'Breakfast'),
    (2, "Coffee")
]

insert_all_new_into("Category", categories)


# %%
curser.execute("select * from product")
res = curser.fetchall()

print(res)
