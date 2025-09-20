
# Install necessary packages
install.packages('arules')
install.packages("pkgbuild")
install.packages("arulesViz")

# Load libraries
library('arules')
library('arulesViz')

# Support 
supp <- function(mydata, premise, implication=list()) {
  count_all <- nrow(mydata)
  conditions <- append(premise, implication)
  for(condition in conditions)
    mydata <- mydata[mydata[condition[1]] == condition[2],]
  return (nrow(mydata) / count_all)
}

# Confidence 
conf <- function(mydata, premise, implication) {
  num <- supp(mydata, premise, implication)
  den <- supp(mydata, premise)
  return (num/den)
}

# Lift 
lift <- function(mydata, premise, implication) {
  num <- conf(mydata, premise, implication)
  den <- supp(mydata, implication)
  return (num/den)
}

# Fairness coefficient
fairness_coeff <- function(mydata, sensitive_attr, non_sensitive_attr1, 
                           target_attr, non_sensitive_attr2=NULL, non_sensitive_attr3=NULL, alpha=1.25) {
  
  if (is.null(non_sensitive_attr2) && is.null(non_sensitive_attr3)) {
    bd <- list(non_sensitive_attr1)
  } else if (!is.null(non_sensitive_attr2) && is.null(non_sensitive_attr3)) {
    bd <- list(non_sensitive_attr1, non_sensitive_attr2)
  } else {
    bd <- list(non_sensitive_attr1, non_sensitive_attr2, non_sensitive_attr3)
  }
  
  val1 <- supp(mydata, bd, list(sensitive_attr))
  val2 <- supp(mydata, list(non_sensitive_attr1), list(sensitive_attr))
  val3 <- conf(mydata, bd, list(sensitive_attr))
  val4 <- conf(mydata, bd, list(target_attr))
  
  # Fairness coefficient formula
  return(((val1 / val2) * (val3 + val4 - 1)) / (val2 * alpha))
}

# Discrimination removal
change_label <- function(data, sensitive_attr, non_sensitive_attr1, target_attr, non_sensitive_attr2=NULL, non_sensitive_attr3=NULL) {
  mydata <- data
  mydata$id <- seq(1, nrow(mydata))  
  
  
  if (target_attr[2] == "TRUE") {
    target <- FALSE
  } else {
    target <- TRUE
  }
  target_attr <- c(target_attr[1], target)
  

  conditions <- list(sensitive_attr, non_sensitive_attr1, target_attr)
  if (!is.null(non_sensitive_attr2)) conditions <- append(conditions, list(non_sensitive_attr2))
  if (!is.null(non_sensitive_attr3)) conditions <- append(conditions, list(non_sensitive_attr3))
  
  # Filtering
  for (condition in conditions) {
    mydata <- mydata[mydata[condition[1]] == condition[2],]
  }
  

  ids <- mydata$id
  if (length(ids) > 0) {
    data[ids[1], target_attr[1]] <- !target
    return(data)
  } else {
    return(NULL)
  }
}


remove_discrimination <- function(data, sensitive_attr, non_sensitive_attr1, target_attr, non_sensitive_attr2=NULL, non_sensitive_attr3=NULL) {
  left <- conf(data, list(non_sensitive_attr1), list(target_attr))
  right <- fairness_coeff(data, sensitive_attr, non_sensitive_attr1, target_attr, non_sensitive_attr2, non_sensitive_attr3)
  
  message("Is ", left, " > ", right, " ?")
  if (left > right) {
    message("Exit loop. No discrimination detected.")
    return(data)
  }
  
  message("Removing detected discrimination...")
  counter <- 0
  while (left <= right) {
    new_data <- change_label(data, sensitive_attr, non_sensitive_attr1, target_attr, non_sensitive_attr2, non_sensitive_attr3)
    if (is.null(new_data)) {
      message("Changed ", counter, " labels.", 
              "New left value is:", left)
      return(data)
    }
    data <- new_data
    left <- conf(data, list(non_sensitive_attr1), list(target_attr))
    counter <- counter + 1
  }
  
  return(data)
}


data <- read.csv("fau_clinic_recruitment.csv")

  # Conditions
sensitive_attr <- c("gender", "m")              # Sensitive attribute
non_sensitive_attr1 <- c("professional", TRUE)  # Non-sensitive attribute 1
target_attr <- c("critical_care_nursing", TRUE) # Target label
non_sensitive_attr2 <- c("empathy", TRUE)       # Non-sensitive attribute 2
non_sensitive_attr3 <- c("patience", TRUE)      # Non-sensitive attribute 3

# Remove discrimination if needed
data <- remove_discrimination(data, sensitive_attr, non_sensitive_attr1, target_attr, non_sensitive_attr2, non_sensitive_attr3)
relevant_data <- subset(data, select = -c(hired, family_nurse, occupational_health_nursing, gerontological_nursing))
rules <- apriori(relevant_data, parameter = list(supp = 0.02, conf = 0.35, target = "rules"),
                 appearance = list(default = "lhs", rhs = "critical_care_nursing"))
inspect(head(rules, n = 20))

subrules <- head(rules, n = 5, by = "lift")
inspect(subrules)

