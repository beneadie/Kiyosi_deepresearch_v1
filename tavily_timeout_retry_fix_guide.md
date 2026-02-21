# Fix Guide: Tavily Search Hangs (Timeout + Retry)

## Problem
In some runs, the pipeline appears to hang around Tavily search.

Typical symptom from logs:
- `Executing Tavily search...`
- long delay
- eventual continuation or manual interruption

Root cause: the Tavily SDK call was synchronous and effectively unbounded in runtime, so the async pipeline could wait far longer than intended.

---

## Files changed
- `deep_research/utils.py`

---

## Root cause in code (before)

`TAVILY_TIMEOUT` existed but was not enforced around the SDK call.

```python
TAVILY_TIMEOUT = 30.0

def tavily_search_multiple(...):
    result = tavily_client.search(...)  # blocking call
```

Also, the Tavily tool called the helper synchronously:

```python
search_results = tavily_search_multiple(...)
```

---

## Implemented fix

### 1) Make helper async

```python
async def tavily_search_multiple(...):
```

### 2) Enforce hard timeout on blocking SDK call

Run Tavily in a thread and wrap with timeout:

```python
result = await asyncio.wait_for(
    asyncio.to_thread(
        tavily_client.search,
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
        days=days,
    ),
    timeout=TAVILY_TIMEOUT,
)
```

### 3) Add retries with backoff

- `max_retries = 3`
- Backoff: `2s`, then `4s`
- On final failure: log error and append empty result (`{"results": []}`)

### 4) Update callsite to await helper

```python
search_results = await tavily_search_multiple(...)
```

---

## Exact behavioral change

### Before
- A slow/stuck Tavily request could block for an unpredictable duration.
- Pipeline could look hung indefinitely at search stage.

### After
- Each Tavily attempt is capped by `TAVILY_TIMEOUT`.
- Failures/timeouts retry automatically up to 3 attempts.
- If all attempts fail, pipeline continues safely with empty results instead of stalling.

---

## Validation checklist

- [ ] `tavily_search_multiple` is `async def`
- [ ] Tavily call wrapped in `asyncio.to_thread(...)`
- [ ] Tavily call wrapped in `asyncio.wait_for(..., timeout=TAVILY_TIMEOUT)`
- [ ] Retry loop (`max_retries = 3`) exists
- [ ] `tavily_search` uses `await tavily_search_multiple(...)`
- [ ] On repeated failure, function returns `{"results": []}` and continues

---

## Expected logs after fix

You should now see bounded behavior like:
- `Tavily search timeout ... (attempt 1/3). Retrying in 2s...`
- `Tavily search timeout ... (attempt 2/3). Retrying in 4s...`
- either success log, or final timeout/failure log and continuation

No indefinite wait at Tavily search stage.

---

## Notes for production agents

- Keep this fix minimal and local to `utils.py`.
- Do not change tool outputs/schema.
- Optional follow-up (separate change): skip summarization for clearly non-article giant payloads (e.g. `.diff`) to reduce long summarization phases that can also look like hangs.
