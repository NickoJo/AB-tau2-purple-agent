import json
import logging
import os
import re
from openai import AsyncOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

logger = logging.getLogger(__name__)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openai.bothub.ru/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")

WINDOW_SIZE = 30


def remove_scratchpad(text: str) -> str:
    """Strip internal reasoning blocks before parsing."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def isolate_json(text: str) -> str:
    """Pull out the JSON payload from a raw LLM response."""
    text = remove_scratchpad(text)

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    return text.strip()


def decode_response(text: str) -> tuple[dict, bool]:
    """Turn raw LLM output into an action dict. Second value signals whether a fallback was used."""
    raw = isolate_json(text)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            return parsed, False
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    logger.warning(f"Response could not be decoded as JSON, using fallback: {text[:200]}")
    return {"name": "respond", "arguments": {"content": text}}, True


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.history: dict[str, list[dict]] = {}
        self.available_tools: dict[str, set[str]] = {}
        self.active_domain: dict[str, str] = {}
        self.client = AsyncOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )

    def _classify_domain(self, opening_message: str) -> str:
        """Infer which service domain is active from the opening policy message."""
        text = opening_message.lower()
        if any(w in text for w in ("airline", "flight", "airport", "reservation")):
            return "airline"
        if any(w in text for w in ("retail", "order", "product", "shopping")):
            return "retail"
        if any(w in text for w in ("telecom", "phone line", "data plan", "suspension", "mobile")):
            return "telecom"
        return "generic"

    def _compose_prompt(self, domain: str) -> str:
        """Return an instruction prompt tailored to the active service domain."""

        shared_footer = (
            "\n## OUTPUT FORMAT\n"
            "Respond with a single raw JSON object only — no prose, no markdown fences.\n"
            "  Tool invocation : {\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}\n"
            "  Message to user : {\"name\": \"respond\", \"arguments\": {\"content\": \"...\"}}"
        )

        shared_rules = (
            "- Treat the policy delivered in the first message as absolute — it overrides everything else.\n"
            "- Issue one tool call per turn; never bundle multiple calls together.\n"
            "- Always wait for an explicit 'yes' from the user before committing any change to the system.\n"
            "- As soon as the user confirms, act immediately — no extra clarification rounds.\n"
            "- Look up any IDs, codes, or references yourself; never ask the user to supply them.\n"
            "- Do not re-request information that already appears in a previous tool result.\n"
            "- Keep the number of tool calls as low as possible to avoid exhausting the turn budget."
        )

        if domain == "airline":
            return (
                "You are an airline support agent operating under a strict policy rulebook.\n\n"
                "## STANDING RULES\n"
                + shared_rules + "\n\n"
                "## CITY → AIRPORT MAPPING (resolve internally — never ask the customer)\n"
                "New York→JFK  Los Angeles→LAX  Chicago→ORD  San Francisco→SFO  Miami→MIA\n"
                "Dallas→DFW  Atlanta→ATL  Seattle→SEA  Boston→BOS  Denver→DEN\n"
                "Houston→IAH  Washington DC→DCA  Philadelphia→PHL  Phoenix→PHX\n"
                "Minneapolis→MSP  Detroit→DTW  Orlando→MCO  Portland→PDX\n"
                "Las Vegas→LAS  Salt Lake City→SLC  Tampa→TPA\n\n"
                "## COMPENSATION RULES (high-error area)\n"
                "- Do not mention compensation unless the customer brings it up first.\n"
                "- Only eligible groups: silver or gold members, passengers with travel insurance, business-class travellers.\n"
                "- Economy / basic-economy passengers with no insurance receive nothing.\n"
                "- Airline-cancelled flight: $100 per passenger in the booking.\n"
                "- Delayed flight: $50 per passenger — but only when the customer is also changing or cancelling; "
                "present this offer after that action is complete, not before.\n"
                "- No other compensation scenarios exist.\n\n"
                "## CANCELLATION ELIGIBILITY\n"
                "- Permitted when: booking made within last 24 h, airline cancelled the flight, "
                "passenger is in business class, or passenger holds travel insurance covering the stated reason (medical / weather).\n"
                "- If any leg of the itinerary has already been flown, cancellation is impossible — escalate to a human agent.\n"
                "- The system will accept any cancel call; eligibility is your responsibility to check first.\n\n"
                "## FLIGHT CHANGES\n"
                "- Basic-economy fares: flight dates and times cannot be changed.\n"
                "- Basic-economy cabin can be upgraded; after upgrading the new cabin's change rules apply.\n"
                "- Cabin cannot change once any flight is flown.\n"
                "- Origin, destination, and trip type are fixed. For a different destination: cancel and rebook.\n"
                "- All flights in one booking must share the same cabin class.\n\n"
                "## NEW BOOKINGS\n"
                "- Hard limit of 5 passengers per booking.\n"
                "- Payment ceiling: 1 travel certificate, 1 credit card, up to 3 gift cards.\n"
                "- Every payment instrument must already exist in the customer's profile.\n"
                "- Ask whether the customer wants travel insurance ($30 per passenger) before finalising.\n\n"
                "## FREE CHECKED BAGS (tier × cabin)\n"
                "- Standard : basic_economy 0 / economy 1 / business 2\n"
                "- Silver    : basic_economy 1 / economy 2 / business 3\n"
                "- Gold      : basic_economy 2 / economy 3 / business 4\n"
                "- Each additional bag costs $50.\n\n"
                "## HUMAN ESCALATION\n"
                "- Escalate only when no available tool can address the request.\n"
                "- Partially-flown booking + cancellation request → escalate.\n"
                "- Do not escalate if you can resolve it yourself."
                + shared_footer
            )

        if domain == "retail":
            return (
                "You are a retail support agent operating under a strict policy rulebook.\n\n"
                "## STANDING RULES\n"
                + shared_rules + "\n\n"
                "## IDENTITY VERIFICATION\n"
                "- Verify the customer's identity before touching any account data.\n"
                "- Accepted methods: email address, or full legal name combined with postal code.\n"
                "- Nothing proceeds until verification is confirmed.\n\n"
                "## ORDER CANCELLATIONS\n"
                "- Only pending orders may be cancelled; shipped or delivered orders cannot.\n"
                "- Confirm the customer's intent before executing the cancellation.\n\n"
                "## ORDER MODIFICATIONS\n"
                "- Only pending orders qualify for modification.\n"
                "- Each order may be modified exactly once.\n"
                "- Permitted changes: delivery address, payment method, item attributes (size, colour, etc.).\n"
                "- The product type itself cannot be swapped (e.g. a jacket cannot become a bag).\n"
                "- Summarise all intended changes and get a clear 'yes' before submitting.\n\n"
                "## RETURNS AND EXCHANGES\n"
                "- Apply the eligibility rules from the policy (return window, item condition, etc.).\n"
                "- The replacement item must be the same product type as the original.\n"
                "- Confirm before lodging the return or exchange.\n\n"
                "## REFUND TIMELINES\n"
                "- Original payment was a gift card → credit appears immediately.\n"
                "- Original payment was a credit card or bank transfer → allow 5–7 business days.\n"
                "- Always tell the customer which timeline applies to them.\n\n"
                "## HUMAN ESCALATION\n"
                "- Escalate only when no available tool can address the request.\n"
                "- Do not escalate if you can resolve it yourself."
                + shared_footer
            )

        if domain == "telecom":
            return (
                "You are a telecom support agent operating under a strict policy rulebook.\n\n"
                "## STANDING RULES\n"
                + shared_rules + "\n\n"
                "## PERMITTED REQUEST TYPES (handle anything else by escalating to a human)\n"
                "  1. Technical troubleshooting\n"
                "  2. Overdue bill payment\n"
                "  3. Line suspension or restoration\n"
                "  4. Service plan changes\n"
                "  5. Mobile data top-up\n\n"
                "## BILLING PAYMENTS\n"
                "- Check the due date before processing; only bills that are already overdue may be paid through this channel.\n"
                "- Reject payment requests for invoices that are not yet due.\n\n"
                "## DATA TOP-UP\n"
                "- Cap each top-up at 2 GB regardless of what the customer requests.\n\n"
                "## LINE RESTORATION\n"
                "- A suspended line cannot be restored if the contract end date has already passed, even when all outstanding bills are cleared.\n"
                "- Check the contract end date before attempting any restoration.\n\n"
                "## HUMAN ESCALATION\n"
                "- Route to a human agent for any request that falls outside the five permitted types.\n"
                "- Do not escalate if you can resolve it yourself."
                + shared_footer
            )

        # generic fallback
        return (
            "You are a customer service agent bound by a strict policy.\n\n"
            "## STANDING RULES\n"
            + shared_rules + "\n"
            "- Deny any request that the policy does not explicitly permit, and cite the relevant rule."
            + shared_footer
        )

    def _periodic_hint(self, domain: str) -> str:
        """Short mid-conversation policy nudge, scoped to the active domain."""
        if domain == "airline":
            return (
                "Policy check-in — keep these in mind:\n"
                "  • Compensation only when the customer asks; eligible tiers: silver/gold, insured, or business.\n"
                "  • Cancelled-flight payout: $100/passenger. Delayed-flight payout: $50/passenger (after change/cancel only).\n"
                "  • Cancellation requires: booking <24 h old, airline-initiated cancel, business class, or insured + covered reason.\n"
                "  • Basic economy: no flight changes allowed, but cabin upgrade is permitted.\n"
                "  • Confirm with the customer before every write operation, then act immediately on 'yes'.\n"
                "  • Never ask for booking reference numbers — retrieve them yourself."
            )
        if domain == "retail":
            return (
                "Policy check-in — keep these in mind:\n"
                "  • Verify identity before any account action.\n"
                "  • Only pending orders can be cancelled or modified; one modification per order maximum.\n"
                "  • Product type is immutable — only attributes like size or colour may change.\n"
                "  • Gift-card refunds are instant; card/bank refunds take 5–7 business days.\n"
                "  • Summarise changes and wait for an explicit 'yes' before committing.\n"
                "  • Look up order details yourself; never ask for order IDs."
            )
        if domain == "telecom":
            return (
                "Policy check-in — keep these in mind:\n"
                "  • Accepted request types: tech support, overdue bills, suspension/restoration, plan changes, data top-up.\n"
                "  • Escalate everything outside that list to a human agent.\n"
                "  • Only process bill payment if the bill is already overdue — check the due date.\n"
                "  • Top-up ceiling is 2 GB per request.\n"
                "  • Restoration is blocked when the contract end date is in the past.\n"
                "  • Confirm before every account change."
            )
        return (
            "Policy check-in:\n"
            "  • Re-read the policy from the first message before your next action.\n"
            "  • One tool call at a time; wait for explicit user confirmation before writing.\n"
            "  • Act immediately once confirmed — no extra back-and-forth."
        )

    def _scan_tools(self, opening_message: str) -> set[str]:
        """Collect tool names declared in the opening policy message."""
        names = {"respond"}
        try:
            for hit in re.finditer(r'"name"\s*:\s*"([^"]+)"', opening_message):
                names.add(hit.group(1))
        except Exception:
            pass
        return names

    def _context_window(self, context_id: str) -> list[dict]:
        """Trim conversation history to stay within the model's effective window."""
        msgs = self.history[context_id]
        if len(msgs) <= WINDOW_SIZE:
            return msgs
        head = msgs[:2]
        head.append({"role": "user", "content": "[Older messages trimmed. Continuing from here.]"})
        tail = msgs[-(WINDOW_SIZE - 3):]
        return head + tail

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        user_input = get_message_text(message)
        context_id = (
            getattr(message, "context_id", None)
            or getattr(message, "contextId", None)
            or "default"
        )

        await updater.update_status(TaskState.working, new_agent_text_message("Working on it..."))

        if context_id not in self.history:
            self.history[context_id] = []

            domain = self._classify_domain(user_input)
            self.active_domain[context_id] = domain
            logger.info(f"Domain identified: {domain}")

            self.history[context_id].append({"role": "system", "content": self._compose_prompt(domain)})
            self.history[context_id].append({"role": "user", "content": user_input})

            self.available_tools[context_id] = self._scan_tools(user_input)
            logger.info(f"Tools available: {self.available_tools[context_id]}")
            logger.info(f"Opening message (preview): {user_input[:500]}")
        else:
            self.history[context_id].append({"role": "user", "content": user_input})

        turn = len(self.history[context_id])
        if turn > 6 and turn % 10 == 0:
            hint = self._periodic_hint(self.active_domain.get(context_id, "generic"))
            self.history[context_id].append({"role": "system", "content": hint})

        try:
            window = self._context_window(context_id)
            logger.info(f"LLM request: model={AGENT_MODEL}, window={len(window)} msgs (total={len(self.history[context_id])})")
            response = await self.client.chat.completions.create(
                model=AGENT_MODEL,
                messages=window,
                temperature=0,
                max_tokens=4096,
            )
            reply = response.choices[0].message.content or ""
            logger.info(f"LLM response (preview): {reply[:300]}")
        except Exception as exc:
            logger.error(f"LLM request failed: {type(exc).__name__}: {exc}")
            reply = json.dumps({"name": "respond", "arguments": {"content": "Something went wrong on my end. Could you restate your request?"}})

        self.history[context_id].append({"role": "assistant", "content": reply})

        action, used_fallback = decode_response(reply)

        allowed = self.available_tools.get(context_id, set())
        if not used_fallback and allowed and action["name"] not in allowed:
            logger.warning(f"Tool '{action['name']}' not in allowed set {allowed} — falling back.")
            used_fallback = True

        if used_fallback:
            logger.warning("Invalid format detected — requesting a corrected response.")
            self.history[context_id].append({
                "role": "user",
                "content": (
                    "That response wasn't in the required format. "
                    "Reply with a single raw JSON object and nothing else:\n"
                    '{"name": "tool_name", "arguments": {...}}  or  '
                    '{"name": "respond", "arguments": {"content": "..."}}'
                ),
            })
            try:
                window = self._context_window(context_id)
                response = await self.client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=window,
                    temperature=0,
                    max_tokens=4096,
                )
                retry_reply = response.choices[0].message.content or ""
                logger.info(f"Corrected response (preview): {retry_reply[:300]}")
                self.history[context_id].append({"role": "assistant", "content": retry_reply})
                retry_action, retry_fallback = decode_response(retry_reply)
                if not retry_fallback:
                    action = retry_action
            except Exception as exc:
                logger.error(f"Correction request failed: {exc}")

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps(action)))],
            name="response",
        )
