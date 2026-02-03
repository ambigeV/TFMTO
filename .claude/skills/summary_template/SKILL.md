# Algorithm Development Report Template

## 1. Algorithm Overview

### 1.1 Basic Information
- **Algorithm Name**: [Name and version]
- **Base Framework**: [e.g., CMA-ES, DE, PSO, etc.]
- **Development Iteration**: [Iteration number in the sequence]
- **Development Date**: [Date]

### 1.2 Design Motivation
- **Problem Context**: [What problems or limitations motivated this variant?]
- **Core Hypothesis**: [What improvement hypothesis was being tested?]
- **Expected Benefits**: [What advantages were anticipated?]

---

## 2. Technical Design

### 2.1 Key Mechanisms
List and explain each core component:

1. **[Mechanism 1 Name]**
   - Description: [How it works]
   - Implementation: [Key technical details]
   - Purpose: [Why this mechanism was added]

2. **[Mechanism 2 Name]**
   - Description: [How it works]
   - Implementation: [Key technical details]
   - Purpose: [Why this mechanism was added]

[Continue for all mechanisms...]

### 2.2 Algorithm Pseudocode
```
[High-level pseudocode or algorithmic flow]
```

### 2.3 Parameter Settings
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Population size | [value] | [reason] |
| [Parameter 2] | [value] | [reason] |
| [Parameter 3] | [value] | [reason] |

---

## 3. Experimental Setup

### 3.1 Benchmark Configuration
- **Benchmark Suite**: [e.g., STSOtest]
- **Problem Set**: [e.g., P1-P9]
- **Dimension**: [e.g., 50]
- **Population Size**: [e.g., 100]
- **Max Evaluations**: [e.g., 50,000]
- **Independent Runs**: [e.g., 10]

### 3.2 Comparison Baselines
| Algorithm | Type | Reason for Inclusion |
|-----------|------|---------------------|
| [Baseline 1] | [e.g., canonical] | [why selected] |
| [Baseline 2] | [e.g., variant] | [why selected] |

---

## 4. Performance Results

### 4.1 Overall Ranking
| Rank | Algorithm | Avg Rank | Performance Tier |
|------|-----------|----------|------------------|
| 1 | [Algorithm] | [value] | Top |
| 2 | [Algorithm] | [value] | Top |
| ... | ... | ... | ... |
| **X** | **[This Algorithm]** | **[value]** | **[tier]** |

### 4.2 Problem-by-Problem Performance
| Problem | This Algorithm | Best Baseline | Winner | Rank |
|---------|----------------|---------------|--------|------|
| P1 ([Name]) | [fitness±std] | [baseline: fitness] | [winner] | [rank] |
| P2 ([Name]) | [fitness±std] | [baseline: fitness] | [winner] | [rank] |
| ... | ... | ... | ... | ... |

### 4.3 Statistical Significance
- **Wilcoxon Test Results**: [Summary of statistical comparisons]
- **Win/Tie/Loss vs Best Baseline**: [X/Y/Z]

---

## 5. Analysis

### 5.1 Strengths
1. **[Strength 1]**: [Evidence from results]
2. **[Strength 2]**: [Evidence from results]
3. **[Strength 3]**: [Evidence from results]

### 5.2 Weaknesses
1. **[Weakness 1]**: [Evidence from results]
2. **[Weakness 2]**: [Evidence from results]
3. **[Weakness 3]**: [Evidence from results]

### 5.3 Mechanism Effectiveness Analysis

#### What Worked
| Mechanism | Evidence | Impact |
|-----------|----------|--------|
| [Mechanism A] | [Performance on specific problems] | [High/Medium/Low] |
| [Mechanism B] | [Performance on specific problems] | [High/Medium/Low] |

#### What Did Not Work
| Mechanism | Evidence | Root Cause |
|-----------|----------|------------|
| [Mechanism X] | [Performance degradation] | [Explanation] |
| [Mechanism Y] | [Performance degradation] | [Explanation] |

### 5.4 Problem Landscape Analysis
- **Separable vs Non-Separable**: [Performance comparison]
- **Unimodal vs Multimodal**: [Performance comparison]
- **Smooth vs Rugged**: [Performance comparison]

---

## 6. Key Insights

### 6.1 Validated Design Principles
1. **[Principle 1]**: [What was learned and confirmed]
2. **[Principle 2]**: [What was learned and confirmed]
3. **[Principle 3]**: [What was learned and confirmed]

### 6.2 Rejected Design Choices
1. **[Choice 1]**: [Why it failed and what to avoid]
2. **[Choice 2]**: [Why it failed and what to avoid]
3. **[Choice 3]**: [Why it failed and what to avoid]

### 6.3 Unexpected Findings
- **[Finding 1]**: [Description and implications]
- **[Finding 2]**: [Description and implications]

---

## 7. Recommendations

### 7.1 For Next Iteration
**Keep**:
- [Mechanism/approach to retain]
- [Mechanism/approach to retain]

**Remove**:
- [Mechanism/approach to discard]
- [Mechanism/approach to discard]

**Explore**:
- [New direction/mechanism to test]
- [New direction/mechanism to test]

### 7.2 Open Questions
1. [Question about mechanism X]
2. [Question about parameter tuning]
3. [Question about problem-specific behavior]

---

## 8. Comparative Evolution

### 8.1 Improvement Over Previous Iteration
- **Previous Variant**: [Name and rank]
- **Current Variant**: [Name and rank]
- **Delta**: [Improvement/degradation]
- **Main Changes**: [What was modified]

### 8.2 Distance from Best Baseline
- **Best Baseline**: [Name and rank]
- **Performance Gap**: [Quantitative difference]
- **Remaining Challenges**: [What prevents matching/beating the baseline]

---

## 9. Reproducibility

### 9.1 Code Availability
- **Implementation**: [Language, framework]
- **Repository**: [Link or location]
- **Dependencies**: [Required libraries/packages]

### 9.2 Random Seed Settings
- **Seed Control**: [How randomness was managed]
- **Reproducibility Verified**: [Yes/No]

---

## 10. Conclusion

### 10.1 Summary
[2-3 sentence summary of the algorithm's performance and position in the development sequence]

### 10.2 Contribution to Development Process
[How this iteration contributed to understanding and improving the algorithm family]

### 10.3 Next Steps
[Concrete action items for the next development cycle]


---

## Usage Instructions

This template is designed for systematic documentation of evolutionary algorithm development. Fill in each section after completing an algorithm iteration:

1. **Sections 1-3**: Complete before experimentation (design documentation)
2. **Section 4**: Fill after running experiments (results)
3. **Sections 5-7**: Complete during analysis phase (insights and learning)
4. **Sections 8-10**: Complete at the end (meta-analysis and planning)

### Tips for Effective Use

- **Be specific**: Use concrete numbers and examples rather than vague descriptions
- **Be honest**: Document what didn't work as thoroughly as what did
- **Be comparative**: Always contextualize performance relative to baselines
- **Be forward-looking**: Use insights to guide the next iteration

### Example Entries

**Good**: "Mirroring sampling reduced variance on P8 (Sphere) by 89%, improving from 1.20e-18 to 1.03e-19"

**Bad**: "Mirroring sampling worked well"

---

## Template Version
- **Version**: 1.0
- **Based on**: LLM-Driven EA Development (LLMEAtest1-10 series)
- **Last Updated**: January 2026