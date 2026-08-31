# Message templates

The wording the bot sends is editable without a code change. Templates live in
the `message_templates` table and are managed through the `/templates` endpoints.

## How it works

There are two kinds of reply, and templates cover both.

**Replies built from data** — a product, an order, a payment receipt. The bot
takes the data the tool returned and renders *your* template with it. The
model's own phrasing is discarded, so these messages look exactly how you set
them up, every time.

**Free-text answers** — anything the model wrote itself. These pass through the
`outbound_wrapper` template, which is just `{message}` by default. Change it if
you want a shop name or signature on every reply.

## Where a template comes from

For any message type, the bot uses the first of these it finds:

1. A row for **your seller id** — your customisation.
2. A row with **no seller id** — a default for every seller.
3. The **built-in default** shipped with the app.

So the table can be completely empty and the bot still works.

## Placeholders

Write `{name}` and it gets filled in. Every template can use:

| Placeholder | Value |
|---|---|
| `{shop_name}` | Your shop name from `seller_profiles.shop_name` |
| `{customer_name}` | The customer's name, when known |
| `{customer_name_suffix}` | `" Nimal"` or `""` — lets one template read "Hello Nimal!" or "Hello!" |

Each message type has its own extras — `{order_id}`, `{total_amount}`,
`{price}`, `{items}` and so on. `GET /templates/keys` lists them per type.

Two things worth knowing:

- **A typo renders as nothing, it does not break the message.** `{custmer_name}`
  becomes empty text. Use the preview endpoint to catch these.
- **Only bare names work.** `{order_id}` yes; `{order.id}`, `{items[0]}` and
  `{price:.2f}` are left as literal text. This is deliberate — templates are
  edited by staff, and allowing attribute access would let a template read
  internals it has no business seeing.

## Message types

| Key | When it's sent |
|---|---|
| `outbound_wrapper` | Wraps every reply no other template matched. **Must contain `{message}`.** |
| `greeting` | First reply in a new conversation |
| `fallback` | The bot could not work out what the customer wants |
| `error` | Something went wrong on our side |
| `handoff` | The conversation was handed to a person |
| `rate_limited` | The customer is sending faster than we can answer |
| `away` | Out of hours (only used if you wire up a schedule) |
| `product_details` | One product |
| `product_list` | The catalogue |
| `product_not_found` | A product search found nothing |
| `order_confirmation` | Right after an order is created |
| `order_details` | An existing order |
| `order_cancelled` | An order was cancelled |
| `tracking_status` | Delivery status |
| `payment_request` | Bank transfer instructions — **put your account details here** |
| `payment_confirmed` | Receipt verified and the amount matched |
| `payment_pending_review` | Receipt saved but unreadable, needs a human |
| `payment_mismatch` | Receipt amount does not match the order total |
| `image_received` | Immediate acknowledgement of a photo |
| `customer_info` | The customer's stored profile |
| `details_request` | Asking a new customer for their details |

## Using the API

**List everything, with the effective text and where it came from:**

```bash
curl "http://localhost:8001/templates?seller_id=7"
```

`source` is `seller`, `global`, or `default`.

**Check wording before saving:**

```bash
curl -X POST http://localhost:8001/templates/greeting/preview \
  -H "Content-Type: application/json" \
  -d '{"body": "Hi{customer_name_suffix}! Welcome to {shop_name} 🌿"}'
```

Returns the rendered text with sample data.

**Save it for one seller:**

```bash
curl -X PUT http://localhost:8001/templates/greeting \
  -H "Content-Type: application/json" \
  -d '{"seller_id": "7", "body": "Hi{customer_name_suffix}! Welcome to {shop_name} 🌿"}'
```

The response includes `unknown_placeholders` — if that list is not empty, you
have a typo.

**Save it for every seller** — omit `seller_id`.

**Revert to the built-in default:**

```bash
curl -X DELETE "http://localhost:8001/templates/greeting?seller_id=7"
```

**After editing the table directly in SQL**, clear the cache:

```bash
curl -X POST http://localhost:8001/templates/reload
```

Templates are cached for 5 minutes, so an edit through the API is immediate but
a direct database change needs this.

## A worked example: bank details

`payment_request` is the one most shops need to change, because the built-in
version has no account number in it.

```bash
curl -X PUT http://localhost:8001/templates/payment_request \
  -H "Content-Type: application/json" \
  -d '{
    "seller_id": "7",
    "body": "💳 *PAYMENT*\n━━━━━━━━━━━━━━━━━━━━\nOrder *#{order_id}* — total *Rs.{total_amount}*\n\nBank: Sampath Bank\nAccount: 1234 5678 9012\nName: Green Leaf Traders\nBranch: Colombo 03\n\nPlease send a photo of your transfer slip once paid. 📸"
  }'
```

## Notes

- WhatsApp uses `*bold*`, `_italic_`, `~strikethrough~` — not markdown `**`.
- A reply longer than 4000 characters is split into several messages
  automatically, so a long `{items}` list is safe.
- If you blank a template, the bot falls back to the model's own words rather
  than sending nothing.
- `outbound_wrapper` without `{message}` is rejected by the API, because saving
  it would throw away every reply.
