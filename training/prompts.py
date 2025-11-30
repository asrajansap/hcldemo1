# app/prompts.py
# Strict prompt: ask LLM to return ONLY JSON with the specified schema.
PROMPT_TEMPLATE = """You are an experienced SAP ABAP engineer and debugger.

Analyze the following ST22 short dump JSON. Return ONLY a single valid JSON object (no other text) with the following keys:
- root_cause (string): short root cause description
- technical_analysis (string): multi-line technical reasoning
- suggested_fix_code (string): ABAP code snippet or patch suggestion
- impact_analysis (string): what other processes/modules may be affected
- priority (string): one of "High","Medium","Low"
- confidence (number): 0.0-1.0 estimated confidence score

Example:
{{
  "root_cause": "Null reference on lv_item when dereferencing.",
  "technical_analysis": "The variable lv_item is set to initial in branch X ...",
  "suggested_fix_code": "IF lv_item IS NOT INITIAL. ... ENDIF.",
  "impact_analysis": "This can affect Z_ORDER_CREATE and batch jobs XYZ.",
  "priority": "High",
  "confidence": 0.85
}}

Now analyze the dump JSON below and produce only the JSON described above.

Dump JSON:
{dump}

Code snippet (if present):
{code}
"""

# Example JSON structure for reference (used for logging / tests)
JSON_RESPONSE_EXAMPLE = {
    "root_cause": "string",
    "technical_analysis": "string",
    "suggested_fix_code": "string",
    "impact_analysis": "string",
    "priority": "High",
    "confidence": 0.9
}
