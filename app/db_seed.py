"""
Database Seeder Script
Populates sample Enterprise Customers, Products (with unique SKUs), and Orders
for the Text-to-SQL Copilot Engine.
"""

import asyncio
from sqlalchemy import text
from app.database import AsyncSessionLocal, init_db
from app.models import Customer, Product, Order, OrderItem

async def seed_database():
    await init_db()
    
    async with AsyncSessionLocal() as session:
        # Check if already seeded
        result = await session.execute(text("SELECT count(*) FROM products;"))
        count = result.scalar()
        if count and count > 0:
            print(f"Database already contains {count} products. Skipping seeding.")
            return

        print("Seeding sample enterprise products, customers, and orders...")
        
        # 1. Add Sample Products with SKUs
        p1 = Product(sku="SKU-SRV-101", name="OmniCloud Enterprise Server 1U", category="Hardware", price=2499.00, stock_quantity=45)
        p2 = Product(sku="SKU-SW-202", name="OmniQuery AI Analytics Pro License", category="Software", price=499.00, stock_quantity=500)
        p3 = Product(sku="SKU-SEC-303", name="Hardware Security Key (FIDO2/MFA)", category="Security", price=55.00, stock_quantity=250)
        p4 = Product(sku="SKU-ACC-404", name="Ergonomic Executive Office Chair", category="Furniture", price=350.00, stock_quantity=80)
        session.add_all([p1, p2, p3, p4])
        await session.flush()

        # 2. Add Sample Customers
        c1 = Customer(name="Acme Financial Corp", email="procurement@acmefin.com", country="USA", tier="Enterprise")
        c2 = Customer(name="Bangalore AI Labs", email="admin@bangaloreai.in", country="India", tier="Gold")
        c3 = Customer(name="Dallas Logistics LLC", email="ops@dallaslogistics.com", country="USA", tier="Standard")
        session.add_all([c1, c2, c3])
        await session.flush()

        # 3. Add Sample Orders
        o1 = Order(customer_id=c1.id, status="Completed", total_amount=5497.00)
        o2 = Order(customer_id=c2.id, status="Completed", total_amount=1497.00)
        o3 = Order(customer_id=c3.id, status="Pending", total_amount=350.00)
        session.add_all([o1, o2, o3])
        await session.flush()

        # 4. Add Order Items
        item1 = OrderItem(order_id=o1.id, product_id=p1.id, quantity=2, unit_price=2499.00)
        item2 = OrderItem(order_id=o1.id, product_id=p2.id, quantity=1, unit_price=499.00)
        item3 = OrderItem(order_id=o2.id, product_id=p2.id, quantity=3, unit_price=499.00)
        item4 = OrderItem(order_id=o3.id, product_id=p4.id, quantity=1, unit_price=350.00)
        session.add_all([item1, item2, item3, item4])

        await session.commit()
        print("✅ Sample database records successfully seeded!")

if __name__ == "__main__":
    asyncio.run(seed_database())
