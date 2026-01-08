**1. Data Processing & Segmentation**
Cleaning: It cleans dates and converts raw payment data into liquidity metrics.
Shortfall Calculation: It calculates the "shortfall" for each loan—the difference between the scheduled installment and the actual amount paid.
Categorization: Loans are grouped into segments based on credit grade (e.g., "AB", "C") and loan term (36 vs. 60 months) to create monthly time series of total shortfalls.
**2. Copula Modeling (Dependency Analysis)**
The code uses Copulas to model how different loan segments behave relative to one another. Unlike simple correlation, copulas help capture "joint tail risk" (i.e., when one group defaults, others are likely to follow).

Families: It fits and compares three copula types: Frank (general dependency), Clayton (lower-tail/recession risk), and Gumbel (upper-tail/extreme event risk).

Selection: It uses the AIC (Akaike Information Criterion) to determine which model best describes the historical relationship between segments.

**3. Risk Simulation (CFaR & ES)**
Simulation: It runs 200,000 Monte Carlo simulations comparing two scenarios:

Independence: Assuming segments fail randomly and independently.

Gumbel Exchangeable: Assuming segments are linked by extreme tail dependence.

Metrics: It calculates two primary risk measures:

CFaR (Cash Flow at Risk): The maximum shortfall expected at a specific confidence level (e.g., 99%).

ES (Expected Shortfall): The average loss expected in the "worst-case" scenarios beyond the CFaR threshold.

**4. Diagnostics & Visualization**
Heavy-Tail Testing: Includes "Hill tail index" and "Mean Excess" plots to see if the losses follow a normal distribution or a "heavy-tailed" (more dangerous) distribution.

Uplift Analysis: Quantifies the "uplift"—how much higher the risk is when you account for dependencies compared to when you ignore them.
