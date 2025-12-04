# Prompt templates for the Medical Agent

# System prompts for different agent roles

ANALYZER_SYSTEM = """You are a medical data analyzer. Your task is to analyze Electronic Health Records (EHR) and laboratory test results to identify patterns, anomalies, and clinically significant findings.

You should:
1. Carefully examine each piece of medical data
2. Identify relevant clinical patterns
3. Note any abnormal values or concerning trends
4. Provide clear, evidence-based analysis

Always be precise and cite specific data points in your analysis."""

REASONER_SYSTEM = """You are a medical reasoning agent. Your task is to synthesize information from multiple sources including EHR data, lab results, transformer model predictions, and web search results to provide comprehensive clinical reasoning.

You should:
1. Consider all available evidence
2. Apply clinical reasoning principles
3. Identify supporting and contradicting evidence
4. Provide well-reasoned conclusions

Be thorough but concise in your reasoning."""

SYNTHESIZER_SYSTEM = """You are a medical report synthesizer. Your task is to combine analysis results from multiple components (transformer predictions, web search findings, trajectory analysis, clustering results) into a coherent summary.

You should:
1. Integrate findings from all sources
2. Highlight key insights and recommendations
3. Note any areas of uncertainty
4. Provide actionable conclusions

Format your output clearly with sections for each major finding."""

# Prompt templates

ANALYSIS_PROMPT = """Analyze the following medical data:

EHR Records:
{ehr}

Lab Test Results:
{lab_tests}

Please provide a comprehensive analysis including:
1. Key clinical observations
2. Abnormal values identified
3. Potential concerns or areas requiring attention
4. Recommended follow-up actions

Format your response with clear sections using XML tags:
<OBSERVATIONS>Your clinical observations</OBSERVATIONS>
<ABNORMALITIES>Any abnormal values identified</ABNORMALITIES>
<CONCERNS>Potential concerns</CONCERNS>
<RECOMMENDATIONS>Recommended actions</RECOMMENDATIONS>"""

REASONING_PROMPT = """Based on the following information, provide clinical reasoning:

Transformer Prediction:
{prediction}

Web Search Findings:
{search_results}

Original Data Summary:
{data_summary}

Please synthesize this information and provide reasoning for the prediction. Consider:
1. How does the prediction align with established medical knowledge?
2. What evidence supports or contradicts the prediction?
3. What additional factors should be considered?

Format your response with:
<REASONING>Your detailed reasoning</REASONING>
<CONFIDENCE>Your confidence level (LOW/MEDIUM/HIGH)</CONFIDENCE>
<CAVEATS>Any important caveats or limitations</CAVEATS>"""

TRAJECTORY_INTERPRETATION_PROMPT = """Interpret the following single-cell trajectory analysis results:

Trajectory Graph:
{trajectory}

Pseudotime Information:
{pseudotime}

Branch Points:
{branches}

Please provide interpretation including:
1. Key developmental or progression patterns
2. Significant branch points and their implications
3. Temporal ordering insights
4. Biological significance

<INTERPRETATION>Your interpretation</INTERPRETATION>
<KEY_FINDINGS>Key findings</KEY_FINDINGS>"""

CLUSTERING_INTERPRETATION_PROMPT = """Interpret the following clustering analysis results:

Cluster Information:
{clusters}

Quality Metrics:
{metrics}

Marker Features:
{markers}

Please provide interpretation including:
1. Characterization of each cluster
2. Relationships between clusters
3. Key distinguishing features
4. Biological or clinical significance

<CLUSTER_PROFILES>Profiles for each cluster</CLUSTER_PROFILES>
<RELATIONSHIPS>Inter-cluster relationships</RELATIONSHIPS>
<SIGNIFICANCE>Clinical/biological significance</SIGNIFICANCE>"""

SUMMARY_PROMPT = """Synthesize all analysis results into a comprehensive summary:

Analysis Components:
1. Transformer Prediction: {transformer_result}
2. Web Search Reasoning: {reasoning_result}
3. Trajectory Analysis: {trajectory_result}
4. Clustering Analysis: {clustering_result}

Original Input:
- EHR Records: {ehr_count} records
- Lab Tests: {lab_tests_shape}

Please provide a comprehensive summary including:
1. Key findings from each component
2. Overall assessment
3. Confidence in findings
4. Recommended next steps

<SUMMARY>Comprehensive summary</SUMMARY>
<ASSESSMENT>Overall assessment</ASSESSMENT>
<NEXT_STEPS>Recommended next steps</NEXT_STEPS>"""

# Validation prompts

VALIDATION_PROMPT = """Validate the following clinical finding against available evidence:

Finding: {finding}

Supporting Evidence:
{supporting_evidence}

Contradicting Evidence:
{contradicting_evidence}

Please evaluate:
1. Strength of supporting evidence
2. Significance of contradicting evidence
3. Overall validity assessment
4. Confidence level

<VALIDITY>VALID/INVALID/UNCERTAIN</VALIDITY>
<CONFIDENCE>Confidence percentage</CONFIDENCE>
<EXPLANATION>Detailed explanation</EXPLANATION>"""

# Error handling prompts

ERROR_ANALYSIS_PROMPT = """An error occurred during analysis. Please review and suggest recovery:

Error Type: {error_type}
Error Message: {error_message}
Context: {context}

Please provide:
1. Likely cause of the error
2. Suggested recovery action
3. Alternative approaches if primary method fails

<CAUSE>Likely cause</CAUSE>
<RECOVERY>Suggested recovery</RECOVERY>
<ALTERNATIVES>Alternative approaches</ALTERNATIVES>"""

# Dictionary for easy access
PROMPTS = {
    "analysis": ANALYSIS_PROMPT,
    "reasoning": REASONING_PROMPT,
    "trajectory_interpretation": TRAJECTORY_INTERPRETATION_PROMPT,
    "clustering_interpretation": CLUSTERING_INTERPRETATION_PROMPT,
    "summary": SUMMARY_PROMPT,
    "validation": VALIDATION_PROMPT,
    "error_analysis": ERROR_ANALYSIS_PROMPT
}

SYSTEMS = {
    "analyzer": ANALYZER_SYSTEM,
    "reasoner": REASONER_SYSTEM,
    "synthesizer": SYNTHESIZER_SYSTEM
}


def get_prompt(name: str, **kwargs) -> str:
    """Get a formatted prompt template."""
    if name not in PROMPTS:
        raise ValueError(f"Unknown prompt: {name}")
    
    template = PROMPTS[name]
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value))
    
    return template


def get_system(name: str) -> str:
    """Get a system prompt."""
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system: {name}")
    return SYSTEMS[name]

