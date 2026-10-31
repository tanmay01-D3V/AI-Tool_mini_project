'''Configuration for the orders dataset used by the Pandera schema.

The dictionary ``ORDERS_COLUMN_TYPES`` maps each column name to a dictionary that
specifies the semantic type, nullability, and any additional constraints such as
uniqueness or numeric ranges.  This data structure is passed directly to
``cleaner.schema.build_schema``.
'''  # noqa: D400

# The order CSV has 16 columns.  The keys below follow the wording used in the
# project specification (semantic_type, nullable, unique, min, max where
# applicable).

ORDERS_COLUMN_TYPES: dict[str, dict] = {
    "order_id": {
        "semantic_type": "id",
        "nullable": False,
        "unique": True,
    },
    "customer_id": {
        "semantic_type": "id",
        "nullable": False,
    },
    "customer_name": {
        "semantic_type": "person_name",
        "nullable": False,
    },
    "email": {
        "semantic_type": "email",
        "nullable": False,
    },
    "phone": {
        "semantic_type": "phone",
        "nullable": False,
    },
    "signup_date": {
        "semantic_type": "date",
        "nullable": False,
    },
    "order_date": {
        "semantic_type": "date",
        "nullable": False,
    },
    "city": {
        "semantic_type": "city",
        "nullable": False,
    },
    "country": {
        "semantic_type": "country",
        "nullable": False,
    },
    "product_category": {
        "semantic_type": "category",
        "nullable": False,
    },
    "quantity": {
        "semantic_type": "integer",
        "nullable": False,
        "min": 1,
        "max": 100,
    },
    "unit_price": {
        "semantic_type": "money",
        "nullable": False,
    },
    "total_amount": {
        "semantic_type": "money",
        "nullable": False,
    },
    "payment_method": {
        "semantic_type": "category",
        "nullable": False,
    },
    "status": {
        "semantic_type": "category",
        "nullable": False,
    },
    "rating": {
        "semantic_type": "integer",
        "nullable": True,
        "min": 1,
        "max": 5,
    },
}

