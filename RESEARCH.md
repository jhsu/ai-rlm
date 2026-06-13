# AI-RLM Marketing Research

## Goal

Establish the baseline audience strategy for a single-page marketing site for `ai-rlm`, aimed at technical AI buyers and builders:

- AI engineer developers who build and operate AI systems.
- CTOs and engineering/data leaders at AI-first or AI-enabled B2B software companies.
- AI transformation and AI leadership buyers with budget authority.

## Base Personas

### 1. AI Engineer Developer

**Source persona:** `persona:peter-programmer` / Peter Programmer

**Relevant job titles:** Software Engineer, Senior Software Engineer, Software Engineer I/II

**Core profile:** Hands-on technical builder motivated by scalable systems, robust architecture, cloud computing, and AI.

**Challenges:**

- Technical debt.
- System scalability.
- Team collaboration.
- Reliability and uptime pressure.
- Bug resolution speed.

**Business needs:**

- Implement robust architecture.
- Reduce infrastructure costs.
- Build scalable systems that are maintainable by engineering teams.

**Likely watering holes:** GitHub, Stack Overflow, Hacker News, tech conferences, AWS re:Invent.

**How AI-RLM should appeal to them:**

- Lead with concrete developer utility, not executive abstraction.
- Show install/use snippets quickly.
- Explain how RLM helps with large-context reasoning, task decomposition, tool execution, and synthesis.
- Make control, debuggability, and implementation details visible.
- Link directly to GitHub, examples, and docs.

### 2. CTO / Technical AI Company Leader

**Source persona:** `persona:dara-the-data-engineering-leader` / Dara the Data Engineering Leader

**Relevant job titles:** CTO, CIO, VP of Data, senior technical operations/data leaders

**Core profile:** Technical leader responsible for architecture, implementation quality, infrastructure decisions, governance, cross-functional technical alignment, and reliable delivery.

**Challenges:**

- Data security and access control.
- Resource management across technical teams.
- Integration across systems.
- Data quality and infrastructure scalability.
- Reducing technical debt and system complexity.

**Business needs:**

- Scalable architecture.
- Secure, compliant, usable data and AI workflows.
- Operational efficiency.
- Decision-making support.
- Technical strategy that does not create long-term fragility.

**How AI-RLM should appeal to them:**

- Position as a composable technical primitive, not a closed product.
- Emphasize fit with existing AI SDK / TypeScript workflows.
- Highlight model/provider flexibility, architecture control, and operational observability.
- Address build-vs-buy concerns: AI-RLM gives internal teams a reusable pattern without hiding internals.
- Show why recursive decomposition can reduce failure modes in long-context, multi-step AI tasks.

### 3. AI Leadership Buyer

**Source persona:** `persona:enterprise-ai-transformation-buyer` / Enterprise AI Transformation Buyer

**Relevant job titles:** CTO, CIO, Chief Data/AI Officer, VP/Head of AI, Director of Data & AI, VP of Business Transformation

**Core profile:** Senior leader accountable for selecting, piloting, governing, and scaling AI capabilities across business functions.

**Challenges:**

- Crowded AI vendor landscape.
- Proving ROI and avoiding hype-driven purchases.
- Data readiness and disconnected systems.
- Security, privacy, compliance, and auditability.
- Integration with existing enterprise architecture.
- Governance, policies, guardrails, and change management.
- Build vs. buy vs. extend decisions.

**Business needs:**

- Coherent AI strategy.
- Prioritized high-impact use cases.
- Clean and well-permissioned data access.
- Scalable integrations.
- Rapid pilots with clear success criteria.
- Sustainable, maintainable AI systems beyond basic chatbots.

**How AI-RLM should appeal to them:**

- Avoid overclaiming business transformation.
- Tie the technical primitive to measurable outcomes: more reliable analysis, better decomposition, fewer brittle prompts, faster engineering iteration.
- Make governance and auditability part of the story, especially for recursive/tool-using workflows.
- Show how AI-RLM can support pilot-friendly adoption by engineering teams.
- Provide proof points, examples, and evaluation criteria once available.

## GetWhys Research Findings

### What AI Engineers Care About

AI engineers evaluate orchestration and reasoning tooling through four main lenses:

- **Governance and safety:** Guardrails, sensitive data protection, permissions, human approval for high-risk steps, and auditability of prompts, outputs, and actions.
- **Integration fit:** Clean connection to existing workflows, APIs, internal databases, CRMs, ITSM systems, and backend processes.
- **Control over internals:** Ability to choose models, configure tools, manage memory/context, and avoid opaque abstractions.
- **Operational reliability:** Observability, evals, reasoning traces, reduced hallucination risk, failure escalation, latency, and cost control.

Implication for the site:

- The hero should not only say "larger contexts." It should also signal controlled recursive reasoning for real AI engineering workflows.
- The page should include a developer-focused section on control: model choice, tools, code execution, intermediate steps, and debuggability.
- It should include a concrete code example or architecture diagram soon after the hero.

GetWhys Sources
† Synthesized from 15 GetWhys buyer interviews conducted 2024–2026 with decision-makers (individual contributor through vp / director) across Financial Services, Software Development, Hospitals and Health Care, Information Technology & Services, and 7 other industries at organizations ranging from small businesses to enterprises (35–2,100,000 employees).

### What CTOs At AI B2B Companies Prioritize

CTOs and CTO-led buying groups prioritize:

- **Security and safe AI usage:** Sensitive data controls, tenant/data segregation, no unintended model training, retention policies, and security artifacts.
- **Governance and cross-functional approval:** Legal, compliance, privacy, and security are often part of AI tooling decisions.
- **Clear ROI and fail-fast pilots:** Tools need a crisp business case and rapid proof of value.
- **Real-system integration:** Demos are not enough; tools need to work with messy data, real workflows, architecture constraints, and existing systems.
- **Scalability and operability:** They prefer platform-level approaches over one-off agent sprawl.
- **Reliability and accuracy:** Trust in outputs is non-negotiable for operational use cases.
- **Cost clarity:** Token, compute, API, and support costs need to be visible and controllable.
- **Avoiding lock-in:** Flexibility across models and providers matters because the model landscape changes quickly.

Implication for the site:

- CTO messaging should frame AI-RLM as an extensible architecture pattern for reliable AI workflows.
- The site should include sections for "Build on your stack," "Inspect every step," and "Avoid model lock-in."
- Any future enterprise-facing copy should address governance and eval readiness without claiming certifications that do not exist.

GetWhys Sources
† Synthesized from 13 GetWhys buyer interviews conducted 2025–2026 with decision-makers (individual contributor through cxo) across Financial Services, Software Development, Medical Equipment Manufacturing, E-Learning Providers, and 4 other industries at organizations ranging from small businesses to enterprises (400–300,000 employees).

### What AI Leaders Need To Approve Tooling

AI leaders with buying authority need evidence across five areas:

- **Risk controls:** Guardrails, human accountability, escalation paths, and clear boundaries for autonomous behavior.
- **Security and compliance readiness:** Documentation, audit reports, pen test evidence, data residency, encryption, and access controls.
- **Measurable business impact:** Time saved, cycle-time reduction, throughput, productivity, remediation time, or cost optimization.
- **Integration and architectural fit:** SSO, stack compatibility, middleware compatibility, real data readiness, and low-friction implementation.
- **Procurement and governance feasibility:** Formal intake workflows, structured pilots, cross-functional testing, and avoidance of tool sprawl.

Implication for the site:

- The current page can stay developer-first, but should include leadership-oriented proof points lower on the page.
- A future version should include "Where it fits" use cases and pilot evaluation criteria.
- Claims should be specific and restrained: AI-RLM helps teams structure recursive AI workflows; it is not a complete governance platform by itself.

GetWhys Sources
† Synthesized from 25 GetWhys buyer interviews conducted 2024–2026 with decision-makers (individual contributor through cxo) across Financial Services, Software Development, Manufacturing, IT Services and IT Consulting, and 10 other industries at organizations ranging from small businesses to enterprises (15–130,000 employees).

## Positioning Strategy

### Primary Positioning

AI-RLM should be positioned as a developer-first library for building recursive AI workflows that decompose complex tasks, run focused sub-queries or code-backed steps, and synthesize results into higher-quality outputs.

### Supporting Themes

- **Reason through larger problems:** Not just more context, but structured decomposition.
- **Keep engineers in control:** Model, tool, memory, and execution choices remain visible and configurable.
- **Make reasoning inspectable:** Intermediate work should be observable and debuggable.
- **Fit into existing TypeScript AI stacks:** Meet developers where they already work.
- **Support serious AI product teams:** Emphasize reliability, scalability, and architecture rather than demo-only magic.

### Messaging By Persona

| Persona | Primary Question | Site Must Answer |
| --- | --- | --- |
| AI engineer developer | "Can I understand and use this quickly?" | Show code, examples, control, and implementation model. |
| CTO / technical AI leader | "Will this improve our architecture without creating risk?" | Show composability, flexibility, observability, and integration fit. |
| AI leadership buyer | "Can this help our team ship valuable AI safely?" | Show business-relevant use cases, pilotability, risk-aware language, and proof path. |

## Recommended Site Structure

1. **Hero**
   - Developer-first headline about recursive AI workflows.
   - Subhead explaining decomposition, focused model calls, code/tool execution, and synthesis.
   - CTAs: GitHub and examples/docs.

2. **Code / Workflow Preview**
   - Show a minimal snippet or pseudo-flow.
   - Make the product feel concrete within the first screen or just below it.

3. **Why Recursive Language Models**
   - Explain why single-shot prompts fail on complex work.
   - Position recursive decomposition as a practical engineering pattern.

4. **Built For AI Engineering Teams**
   - Cards for control, observability, model flexibility, and integration.

5. **Use Cases**
   - Long-document reasoning.
   - Multi-step research and synthesis.
   - Code/document comparison.
   - Data transformation or extraction workflows.
   - Agent/tool orchestration prototypes.

6. **Leadership Confidence**
   - Explain how inspectable intermediate steps and controlled execution support evaluation, debugging, and pilot readiness.
   - Keep this section grounded and avoid overclaiming enterprise governance.

7. **Final CTA**
   - View GitHub.
   - Run an example.
   - Read docs.

## Proposed Changes To Current Site

The current site is a good shell, but it should evolve in these ways:

- Replace the current generic hero with a sharper developer-first positioning statement.
- Add a code or workflow preview immediately after the hero.
- Replace the current three generic cards with persona-aligned benefit cards: `Decompose complex tasks`, `Inspect intermediate reasoning`, `Run with your models and tools`.
- Add a use-case section that maps to the examples already in the repo.
- Add a leadership confidence section focused on reliability, auditability, and pilot readiness.
- Keep GitHub as the primary CTA until there is product documentation, hosted examples, or a live demo.
- Preserve the simple static implementation for now; no framework is needed until content, routing, or interactivity becomes more complex.

## Open Content Questions

- What is the strongest real-world example we can show first: document comparison, data transformation, long-context research, or tool orchestration?
- Should the site describe AI-RLM as a "library," "framework," "pattern," or "runtime"?
- Is the intended adoption motion open-source developer adoption first, commercial leads first, or both?
- What proof points are available now: benchmarks, example outputs, architecture diagrams, customer/user quotes, or internal demos?
