library(ggplot2)
library(ggsignif)

# Define the function
create_boxplot <- function(data, to_compare, comparisons, summary_stats, max_by_condition, colors, y_offset = 1.5, text_size = 4) {
  # Create the plot
  p <- ggplot(data, aes(x = condition, y = to_compare)) +
    geom_hline(yintercept = seq(0, 20, by = 1), 
               color = "lightgrey", 
               size = 0.2) +
    stat_boxplot(geom = 'errorbar', width = 0.5) +
    geom_boxplot(fill = colors) + # Boxplot fill colors
    
    # Add significance annotation with adjusted y_position
    geom_signif(
      comparisons = comparisons,
      map_signif_level = TRUE,
      y_position = max(max_by_condition$max_value) + y_offset  
    ) +
    
    annotate("text", 
             x = summary_stats$condition, 
             y = max_by_condition$max_value + 1,  # Adjust y position based on max value for each condition
             label = paste("M =", round(summary_stats$median, 2), 
                           "\nSD =", round(summary_stats$sd, 2)), 
             size = text_size,  # Adjust font size
             colour = "black",
             hjust = 0.5) +
    
   
    theme(
      panel.background = element_blank(),  
      axis.title = element_blank(),  
      axis.text.x = element_blank(),  
      axis.ticks.x = element_blank(),    
      panel.grid.major.x = element_blank(), 
      panel.grid.minor.x = element_blank()
    )
  
  print(p)
}