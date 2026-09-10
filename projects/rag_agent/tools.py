import requests

tools = [
    {
        "type": "web_search"
    },
    {
        "type": "function",
        "name": "convert_currency",
        "description": "Convert an amount form one currency to another using real-time exchange rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount":        {"type": "number", "description": "The amount of money to convert"},
                "from_currency": {"type": "string", "description": "The source currency code (e.g, 'USD, 'EUR)"},
                "to_currency":   {"type": "string", "description": "The target currency code (e.g., 'USD', 'EUR')"}
            },
            "required": ["amount", "from_currency", "to_currency"],
            "additionalProperties": False
        }
    },
]


def convert_currency(amount, from_currency, to_currency):
    url = f"https://api.frankfurter.dev/v1/latest?base={from_currency}&symbols={to_currency}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        rate = data["rates"].get(to_currency)
        
        if rate is None:
            return f"Could not find exchange rate for {from_currency} to {to_currency}"
        
        converted_amount = amount * rate
        return f"{amount} {from_currency} = {converted_amount:.2f} {to_currency} (Rate: {rate})"
    except requests.exceptions.RequestException as e:
        return f"Error converting currency: {str(e)}"

