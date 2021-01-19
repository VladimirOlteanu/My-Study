library(shiny)
library(readr)
library(ggplot2)
library(tidyverse)
library(gridExtra)
library(lubridate)
covid19_data<-as.data.frame(read_csv("COVID-19_activity_v2.csv"))
country<-unique(covid19_data$ COUNTRY_SHORT_NAME)
country<-sort(country)
# Define UI for application that draws a scatterplot
ui <- fluidPage(
  
  # Application title
  titlePanel('Covid 19 basic dashboard'),
  
  # Sidebar with a slider input for the country selection 
  
  selectInput('country',
              'Please select a country:',choices=country
  ),
  
  # Show a scatterplot
  mainPanel(
    plotOutput('scatterPlot')
  )
)


# Define server logic required to draw a histogram
server <- function(input, output) {
  
  output$scatterPlot <- renderPlot({
    # select country based on input$Country from ui.R
    
    covid19_country_data<-covid19_data %>% filter(COUNTRY_SHORT_NAME== input$country)%>% 
      mutate(REPORT_DATE = mdy(REPORT_DATE))
    
    # draw scatter plot per the country of choice
    
    plot1<-ggplot(covid19_country_data,aes(x=REPORT_DATE,y=PEOPLE_POSITIVE_CASES_COUNT), color='blue')+
      geom_line()+
      theme_classic()+
      theme(axis.title = element_text(size = 17, face = 'bold'))+
      xlab('Date reported')+
      ylab('Nb. of positive cases')
    
    plot2<-ggplot(covid19_country_data,aes(x=REPORT_DATE,y=PEOPLE_POSITIVE_NEW_CASES_COUNT) , color='blue')+
      geom_line()+
      theme_classic()+
      theme(axis.title = element_text(size = 17, face = 'bold'))+
      xlab('Date reported')+
      ylab('Nb. of new cases')
    
    plot3<-ggplot(covid19_country_data,aes(x=REPORT_DATE,y=PEOPLE_DEATH_COUNT), color='red')+
      geom_line()+
      theme_classic()+
      theme(axis.title = element_text(size = 17, face = 'bold')) +
      xlab('Date reported')+
      ylab('Nb. of deaths')
    
    plot4<-ggplot(covid19_country_data,aes(x=REPORT_DATE,y=PEOPLE_DEATH_NEW_COUNT) , color='red')+
      geom_line()+
      theme_classic()+
      theme(axis.title = element_text(size = 8, face = 'bold')) +
      xlab('Date reported')+
      ylab('Nb. of new deaths')
    
    grid.arrange(plot1, plot2, plot3, plot4, nrow=2, ncol=2)
    
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
