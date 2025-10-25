# HRM-Indian-Lawyer-LLM

Short research README for an Indian lawyer LLM tailored to HRM (employment and workplace law) use-cases. The project pairs a fine-tuned LLM backend with a lightweight web frontend and is trained/evaluated on the ILDC dataset.

## Key points
- Purpose: Legal question-answering, clause drafting, and retrieval for Indian HR/employment law.
- Dataset: ILDC (Indian Legal Documents Corpus) used for supervised fine-tuning and retrieval-index construction. Preprocessing, license and access instructions must follow ILDC terms.
- Model: Transformer-based LLM fine-tuned on ILDC with task adapters for HRM intents and legal style constraints.
- Frontend: Simple React/Vue single-page app for conversational queries, document upload, and citation-aware responses.

## Evaluation & safety
- Evaluate on held-out ILDC splits and human expert review for legal accuracy.
- Include conservatism: disclaimers, citation of sources, and escalation to qualified counsel for critical decisions.

## Responsible use
For research and assistive purposes only. Not a substitute for legal advice. Ensure compliance with local regulations and dataset licensing.

## License & citation
```Note to futureself: Specify project license and cite ILDC and base LLM papers when publishing.```

For implementation details, scripts, and configuration, see the repository folders: data/, model/, frontend/, eval/.