from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# Document type metadata — drives agent behaviour and output structure
# ---------------------------------------------------------------------------
DOC_TYPE_CONTEXT: dict[str, dict] = {
    "PRD": {
        "full_name": "Product Requirements Document",
        "key_sections": "Executive Summary, Problem Statement, Goals & Objectives, User Personas, Features & Requirements, Out of Scope, Success Metrics, Timeline",
        "key_questions": "product vision, target users, core features, success metrics, constraints, timeline",
    },
    "SRS": {
        "full_name": "Software Requirements Specification",
        "key_sections": "Introduction, System Overview, Functional Requirements, Non-Functional Requirements, System Constraints, Interface Requirements, Data Requirements, Appendix",
        "key_questions": "system scope, functional requirements, non-functional requirements (performance, security, scalability), technology stack, integrations",
    },
    "BRD": {
        "full_name": "Business Requirements Document",
        "key_sections": "Executive Summary, Business Objectives, Current State, Proposed Solution, Stakeholders, Business Requirements, Assumptions & Constraints, ROI & Benefits",
        "key_questions": "business problem, stakeholders, current pain points, proposed solution, ROI expectations, budget, timeline",
    },
    "FD": {
        "full_name": "Functional Design Document",
        "key_sections": "Overview, User Flows, Feature Specifications, UI/UX Requirements, Business Rules, Data Model, API Contracts, Error Handling",
        "key_questions": "user flows, feature specifications, business rules, UI requirements, data model, API design",
    },
    "TD": {
        "full_name": "Technical Design Document",
        "key_sections": "Architecture Overview, System Components, Technology Stack, Database Design, API Design, Security Design, Deployment Architecture, Performance Considerations",
        "key_questions": "system architecture, technology choices, database design, API contracts, deployment environment, security requirements, scalability approach",
    },
    "Project Proposal": {
        "full_name": "Project Proposal",
        "key_sections": "Executive Summary, Problem Statement, Proposed Solution, Scope & Deliverables, Timeline & Milestones, Budget Estimate, Team & Responsibilities, Risk Assessment",
        "key_questions": "project goal, scope, key deliverables, timeline, budget, team structure, risks",
    },
}

_BASE = "You are DocMind AI, an expert technical writer and document strategist.\n\n"

# ---------------------------------------------------------------------------
# Requirements agent — gathers info one question at a time
# ---------------------------------------------------------------------------
_REQUIREMENTS_INSTRUCTIONS = """You are gathering requirements to write a {doc_type_full}.

Rules:
- Ask exactly ONE focused question per turn
- Each question uncovers a critical piece of information needed for: {key_sections}
- Focus on: {key_questions}
- After 4–6 exchanges, when you have enough context to write a complete document, respond ONLY with the exact text: REQUIREMENTS_COMPLETE
- Be concise and professional — no preamble, just the question

Do NOT generate the document yet. Do NOT explain what you are doing. Just ask the next most important question."""

requirements_prompt = ChatPromptTemplate.from_messages([
    ("system", _BASE + _REQUIREMENTS_INSTRUCTIONS),
    MessagesPlaceholder(variable_name="messages"),
])

# ---------------------------------------------------------------------------
# Writer agent — generates the full structured document
# ---------------------------------------------------------------------------
_WRITER_INSTRUCTIONS = """Generate a complete, professional {doc_type_full}.

Required sections: {key_sections}

Requirements gathered from the user:
{requirements_summary}
{rag_context_section}
{research_context_section}

Guidelines:
- Use proper markdown: ## for main sections, ### for subsections, **bold** for emphasis
- Be specific and detailed — no vague placeholder text
- Every section must have substantive content
- Write in a professional tone appropriate for the document type
- After completing the document, do NOT ask any questions — output only the document"""

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", _BASE + _WRITER_INSTRUCTIONS),
    ("human", "Generate the complete {doc_type_full} now based on everything discussed."),
])

# ---------------------------------------------------------------------------
# Section regeneration — targeted rewrite of one section
# ---------------------------------------------------------------------------
_SECTION_REGEN_INSTRUCTIONS = """The user wants to modify a specific section of their {doc_type_full}.

Current full document:
{full_document}

Instructions:
- Identify the section the user is referring to
- Rewrite ONLY that section based on the user's instruction
- Return the COMPLETE updated document with the section replaced
- Maintain consistent style, tone, and markdown formatting throughout
- Do NOT add commentary — output only the full updated document"""

section_regen_prompt = ChatPromptTemplate.from_messages([
    ("system", _BASE + _SECTION_REGEN_INSTRUCTIONS),
    ("human", "{user_request}"),
])

# ---------------------------------------------------------------------------
# Document enhancement — for uploaded user documents
# ---------------------------------------------------------------------------
_ENHANCE_INSTRUCTIONS = """The user has uploaded a document. Here is its content:

{document}

You are an expert document enhancement assistant. Based on the user's request you can:

**Fill & Complete**
- Detect placeholders like [NAME], {{field}}, <BLANK>, TBD, or empty sections
- Fill them with contextually appropriate, professional content

**Enhance & Rewrite**
- Rewrite sections to be more professional, formal, persuasive, or clear
- Improve tone, grammar, flow, and structure throughout

**Expand & Add**
- Expand thin sections with high-quality, relevant content
- Add new clauses, sections, or details the user requests
- Insert industry-standard language where appropriate

**Summarize & Explain**
- Summarize the document or any section
- Explain complex clauses in plain language

Always preserve the original document structure unless asked to change it.
When producing an updated document, output the COMPLETE document — not just the changed parts.
After delivering, ask if they want further refinements."""

enhance_prompt = ChatPromptTemplate.from_messages([
    ("system", _BASE + _ENHANCE_INSTRUCTIONS),
    MessagesPlaceholder(variable_name="messages"),
])

# ---------------------------------------------------------------------------
# Research-grounded generation
# ---------------------------------------------------------------------------
_RESEARCH_INSTRUCTIONS = """The user wants a document. Relevant web research has been gathered to ensure accuracy.

Web Research Results:
{research}

Using this research:
- Write a well-informed, professional document grounded in the research above
- Cite sources inline where relevant (e.g. "According to [source]...")
- If the request is still unclear, ask ONE clarifying question before generating
- After generating, ask if they want to refine any section"""

rag_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", _BASE + """You are a knowledge base assistant. Answer the user's question using ONLY the context below.

Context retrieved from the knowledge base:
{context}

Rules:
- Answer directly and concisely based on the context
- If the context doesn't contain enough information, say so clearly
- Quote relevant excerpts when helpful
- Do not make up information not present in the context"""),
    ("human", "{query}"),
])

research_prompt = ChatPromptTemplate.from_messages([
    ("system", _BASE + _RESEARCH_INSTRUCTIONS),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
