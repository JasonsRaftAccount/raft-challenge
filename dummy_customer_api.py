# file: dummy_customer_api.py
from flask import Flask, request, jsonify
from faker import Faker
from faker_commerce import Provider as CommerceProvider
import random

app = Flask(__name__)
fake = Faker("en_US")  # US locale for addresses
fake.add_provider(CommerceProvider)
Faker.seed(42)  # Reproducible data
random.seed(42)


def calculate_return_probability(avg_rating: float) -> float:
    """
    Calculate probability of order return based on average item rating.

    SIMULATED RELATIONSHIP FOR DEMONSTRATION PURPOSES:
    In a real-world setting, this linear relationship would be derived from
    historical order data and industry benchmarks. E-commerce return rates
    typically range from 15-30% overall, with strong correlation to customer
    satisfaction metrics like product ratings.

    This simulation uses a simple linear model:
    - Rating 1.0 → 50% return probability (dissatisfied customer)
    - Rating 5.0 → 5% return probability (satisfied customer)

    Formula: return_prob = 0.50 - (avg_rating - 1.0) * 0.1125
    """
    return_prob = 0.50 - (avg_rating - 1.0) * 0.1125
    return max(0.0, min(1.0, return_prob))  # Clamp to 0-1


def generate_order(order_id: int) -> str:
    """Generate a single order with ratings and return status."""
    # Random buyer
    buyer = fake.name()

    # Random US location
    city = fake.city()
    state = fake.state_abbr()

    # Random items (1-4 items per order)
    num_items = random.randint(1, 4)

    # Calculate total and item ratings
    total = 0.0
    items_with_ratings = []
    ratings = []

    for _ in range(num_items):
        # Generate product name and price using faker-commerce
        product_name = fake.ecommerce_name()
        price = round(random.uniform(9.99, 999.99), 2)  # $9.99 to $999.99
        total += price

        # Random rating 1.0-5.0
        rating = round(random.uniform(1.0, 5.0), 1)
        ratings.append(rating)

        items_with_ratings.append(f"{product_name} ({rating}*)")

    # Average rating for return probability
    avg_rating = sum(ratings) / len(ratings)

    # Calculate return probability using linear relationship
    return_prob = calculate_return_probability(avg_rating)
    returned = "Yes" if random.random() < return_prob else "No"

    # Format order string
    items_str = ", ".join(items_with_ratings)
    order_str = (
        f"Order {order_id}: Buyer={buyer}, Location={city}, {state}, "
        f"Total=${total:.2f}, Items: {items_str}, Returned={returned}"
    )

    return order_str


def generate_orders(count: int = 250) -> list:
    """Generate multiple orders starting from ID 1001."""
    return [generate_order(1001 + i) for i in range(count)]


# Generate 250 orders at startup
ORDERS = generate_orders(250)


@app.route("/api/orders", methods=["GET"])
def get_orders():
    """
    Returns orders as messy text. In real life, customers
    would have unpredictable formatting. The AI must parse it.
    """
    limit = request.args.get("limit", default=len(ORDERS), type=int)
    sample = random.sample(ORDERS, min(limit, len(ORDERS)))

    return jsonify({"status": "ok", "raw_orders": sample})


@app.route("/api/order/<order_id>", methods=["GET"])
def get_order_by_id(order_id):
    """
    Fetch a single order by scanning the text.
    """
    for text in ORDERS:
        if f"Order {order_id}:" in text:
            return jsonify({"status": "ok", "raw_order": text})

    return jsonify({"status": "not_found"}), 404


if __name__ == "__main__":
    print(f"Generated {len(ORDERS)} orders")
    print(f"Sample order: {ORDERS[0]}")
    app.run(host="0.0.0.0", port=5001, debug=True)
