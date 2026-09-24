"""Available checkpoint funding; no implicit exchanges or future balances."""
import math

class NoFeasibleAction(ValueError):
    pass

def funding_choice(state, action, timestamp, price, minimum_usd=50):
    if state is None or state.get("timestamp", math.inf) > timestamp:
        raise NoFeasibleAction("Missing or future checkpoint state")
    candidates = []
    for symbol, balance in state.get("wallet_balances", {}).items():
        amount = float(balance)
        if action.lower() == "repay":
            amount = min(amount, float(state.get("debt_balances", {}).get(symbol, 0)))
        try:
            quote = float(price(symbol))
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(amount) and math.isfinite(quote) and amount > 0 and quote > 0:
            candidates.append((amount * quote, symbol, amount, quote))
    if not candidates:
        raise NoFeasibleAction("No priced, same-asset checkpoint funds for " + action)
    _, symbol, maximum, quote = max(candidates)
    minimum = min(max(minimum_usd / quote, maximum / 12), maximum)
    return symbol, minimum, maximum
