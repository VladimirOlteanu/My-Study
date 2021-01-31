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
    
    plot1<-ggplot(covid19_country_data,aes(x=REPORT_DATE,y=PEOPLE_POSITIVE_CASES_COUNT))+
      geom_line(color='blue')+
      theme_classic()+
      theme(axis.title = element_text(size = 12, face = 'bold'))+
      xlab('Date reported')+
      ylab('Nb. of positive cases')
    
    plot2<-ggplot(covid19_country_data,aes(x=REPORT_DATE,y=PEOPLE_POSITIVE_NEW_CASES_COUNT))+
      geom_line(color='blue')+
      theme_classic()+
      theme(axis.title = element_text(size = 12, face = 'bold'))+
      xlab('Date reported')+
      ylab('Nb. of new cases')
    
    plot3<-ggplot(covid19_country_data, aes(x=PEOPLE_POSITIVE_NEW_CASES_COUNT))+
      geom_density(color='darkblue', fill='lightblue')+
      xlab("Nb. of new cases")+ 
      geom_vline(aes(xintercept=mean(PEOPLE_POSITIVE_NEW_CASES_COUNT)), color='blue', linetype='dashed', size=1)+
      theme(axis.title = element_text(size = 12))+ 
      theme_classic()
    
    plot4<-ggplot(covid19_country_data,aes(x=REPORT_DATE,y=PEOPLE_DEATH_COUNT))+
      geom_line(color='red')+
      theme_classic()+
      theme(axis.title = element_text(size = 12, face = 'bold'))+
      xlab('Date reported')+
      ylab('Nb. of deaths')
    
    plot5<-ggplot(covid19_country_data,aes(x=REPORT_DATE,y=PEOPLE_DEATH_NEW_COUNT))+
      geom_line(color='red')+
      theme_classic()+
      theme(axis.title = element_text(size = 12, face = 'bold'))+
      xlab('Date reported')+
      ylab('Nb. of new deaths')
    
     plot6<-ggplot(covid19_country_data, aes(x=PEOPLE_DEATH_NEW_COUNT))+
      geom_density(color='black', fill='red')+
      xlab('Nb. of new deaths')+
      geom_vline(aes(xintercept=mean(PEOPLE_DEATH_NEW_COUNT)),color='black', linetype='dashed', size=1)+
      theme(text = element_text(size = 12))+
      theme_classic()
      
    
    grid.arrange(plot1, plot2, plot3, plot4, plot5, plot6, nrow=3, ncol=3)
    
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
