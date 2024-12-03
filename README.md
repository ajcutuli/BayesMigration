# A Bayesian Hierarchical Framework for Capturing Preference Heterogeneity in Migration Flows

## Abstract
Understanding and predicting human migration patterns is a central challenge in population dynamics research. 
Traditional physics-inspired gravity and radiation models represent migration flows as functions of attractiveness using socio-economic features as proxies. 
They assume that the relationship between features and migration is spatially invariant, regardless of the origin and destination locations of migrants. 
We use Bayesian hierarchical models to demonstrate that migrant preferences likely vary based on geographical context, specifically the origin-destination pair. 
By applying these models to U.S. interstate migration data, we show that incorporating heterogeneity in a single latent migration parameter significantly improves the ability to explain variations in migrant flows. 
Accounting for such heterogeneity enables it to outperform classical methods and recent machine-learning approaches. 
A clustering analysis of spatially varying parameters reveals two distinct groups of migration paths.
Individuals migrating along low-flow paths (typically between smaller populations or over larger distances) exhibit more nuanced decision-making. 
Their choices are less directly influenced by specific destination characteristics such as housing costs, land area, and climate-related disaster costs. 
High-flow path migrants appear to respond more directly to these destination attributes. 
Our results challenge assumptions of uniform preferences and underscore the value of capturing heterogeneity in migration models and policymaking.

## Code
The code accompanying this paper is fully contained in this repository.
In order to replicate the results, one can first execute **data_cleaning.ipynb**, then run through the various modeling notebooks, and lastly execute **results_analysis.ipynb** to generate the clustering results presented in the paper.