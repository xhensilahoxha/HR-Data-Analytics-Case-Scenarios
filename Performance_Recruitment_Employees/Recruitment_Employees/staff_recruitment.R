# By the use of this code one identifies association rules within the recruitment data, 
# particularly focusing on relationships involving the "critical_care_nursing" position. 
# The aim is to uncover patterns or factors that may influence the recruitment process. 

install.packages('arules')
install.packages("pkgbuild")
install.packages("arulesViz")

library('arules')
library('arulesViz')

data <- read.csv("fau_clinic_recruitment.csv")

rules <- apriori(data, parameter = list(supp = 0.02, conf = 0.5, target = "rules"))
inspect(head(rules, n = 10))
rules <- apriori(data, parameter = list(supp = 0.02, conf = 0.3, target = "rules"),
                 appearance = list(default = "lhs", rhs = "critical_care_nursing"))
inspect(head(rules, n = 10))

relevant_data <- subset(data, select = -c(hired, family_nurse, occupational_health_nursing, gerontological_nursing))
rules <- apriori(relevant_data, parameter = list(supp = 0.02, conf = 0.35, target = "rules"),
                 appearance = list(default = "lhs", rhs = "critical_care_nursing"))
inspect(head(rules, n = 20))

subrules <- head(rules, n = 5, by = "lift")
inspect(subrules)