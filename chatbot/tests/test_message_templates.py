"""
The built-in MessageTemplates, exercised with the exact shapes the repository
functions actually return - which is how the "Unknown Product" bug survived.
"""
from templates.message_templates import MessageTemplates


# The literal return value of repositories.tools.get_product_info
PRODUCT = {
    "product_id": 3,
    "product": "Ceylon Tea 500g",
    "description": "Loose leaf",
    "price": 1250.0,
    "stock": 42,
    "images": [],
}


def test_product_details_shows_the_real_name_and_id():
    """Regression: the template read 'name'/'id' while get_product_info returns
    'product'/'product_id', so every product was shown to the customer as
    "Product ID: N/A / Name: Unknown Product"."""
    out = MessageTemplates.product_details(PRODUCT)
    assert "Ceylon Tea 500g" in out
    assert "Unknown Product" not in out
    assert "N/A" not in out
    assert "3" in out


def test_product_details_still_accepts_the_other_spelling():
    out = MessageTemplates.product_details(
        {"id": 9, "name": "Spice Box", "description": "d", "price": 890.0, "stock": 5}
    )
    assert "Spice Box" in out
    assert "Unknown Product" not in out


def test_product_details_falls_back_when_the_name_is_missing():
    out = MessageTemplates.product_details({"price": 10, "stock": 1})
    assert "Unknown Product" in out


def test_product_details_accepts_the_string_form():
    out = MessageTemplates.product_details(
        "Product ID: 1, Product: Tea, Description: nice, Price: Rs.100, Stock: 5"
    )
    assert "Tea" in out


def test_order_details_renders_items():
    out = MessageTemplates.order_details(
        {
            "order_id": 12,
            "status": "pending",
            "total_amount": 2140.0,
            "items": [{"product": "Tea", "quantity": 2, "price": 1070.0}],
        }
    )
    assert "12" in out and "Tea" in out


def test_product_list_renders_each_product():
    out = MessageTemplates.product_list(
        [{"name": "Tea", "price": 1250.0, "stock": 42}, {"name": "Spices", "price": 890.0, "stock": 3}]
    )
    assert "Tea" in out and "Spices" in out


def test_customer_info_accepts_the_string_get_user_info_returns():
    out = MessageTemplates.customer_info(
        "User ID: 947, Name: Nimal, Email: n@x.com, Address: Colombo, Phone: +94771234567"
    )
    assert "Nimal" in out


def test_templates_never_raise_on_odd_input():
    """These run inside tool wrappers; an exception here loses the reply."""
    for value in (None, {}, [], "", 0):
        assert MessageTemplates.product_details(value)
        assert MessageTemplates.order_details(value)
        assert MessageTemplates.product_list(value)
        assert MessageTemplates.customer_info(value)
