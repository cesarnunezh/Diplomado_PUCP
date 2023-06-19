# MI PRIMERA APLICACI??N R SHINY -----

# 0. Importing and calling libraries ----
library(tidyverse)
library(foreign)
library(shiny)
library(labelled)
library(survey)
library(shiny)
library(shinyWidgets)
library(shinythemes)
library(shinycssloaders)
library(mapsPERU)
library(sf)
library(scales)


# 1. Importing the data ----
## First we set the working directory
setwd("D:/1. Documentos/0. Bases de datos/2. ENAHO/1. Data")

## Now we import the dataset that it is in a STATA format
for (i in 2017:2021) {
  
  # create file name using paste0 function
  file_name <- paste0("https://github.com/cesarnunezh/CNHGitHub/raw/main/Trabajo%20final%20R/enaho01-", i, "-100.dta")
  
  alternative <- read.dta(file_name) |> 
    select('ubigeo','estrato','factor07','p110', 'p110c1', 'p111a', 'p1121', 'p1142', 'p1144', 'mes', 'conglome', 'vivienda', 'hogar', 'dominio') |> 
    mutate(anio=i)
  
  # read the data into R using read.dta function
  assign(paste0("data_", i), alternative)
  
}

# 2. PREPROCESAMIENTO ------
# Todas las acciones que debe ejecutar el programa antes de llegar al UI y server.

  ## Now we manipulate the data


  df_names <- c("data_2017", "data_2018", "data_2019", "data_2020", "data_2021")
  years <- c(2017, 2018, 2019, 2020, 2021)
  
  
  for (df_name in df_names) {
  
    df <- get(df_name)
    
    df <- df |> 
      mutate(ubigeo_n = as.numeric(ubigeo)) |> 
      mutate(dpto = round(ubigeo_n/10000)) |> 
      mutate(depto = case_when(dpto == 1 ~ "Amazonas", dpto == 2 ~ "Ancash", dpto == 3 ~ "Apurimac", dpto == 4 ~"Arequipa", dpto ==  5 ~ "Ayacucho", dpto ==  6 ~ "Cajamarca",
                               dpto == 7 ~ "Callao", dpto == 8 ~ "Cusco", dpto == 9 ~ "Huancavelica", dpto == 10 ~ "Hu??nuco", dpto == 11~ "Ica", dpto == 12 ~ "Jun??n",
                               dpto ==13 ~ "La Libertad", dpto ==14 ~ "Lambayeque", dpto == 15 ~ "Lima", dpto == 16 ~ "Loreto", dpto == 17 ~ "Madre de Dios", dpto == 18 ~ "Moquegua",
                               dpto == 19 ~ "Pasco", dpto == 20 ~ "Piura", dpto == 21 ~ "Puno", dpto == 22 ~ "San Mart??n", dpto == 23 ~ "Tacna", dpto == 24 ~ "Tumbes", dpto == 25 ~ "Ucayali")) |> 
      mutate(area = case_when(estrato == " \xc1rea de empadronamiento rural (aer) simple" ~ "Rural",
                              estrato == " \xc1rea de empadronamiento rural (aer) compuesto" ~ "Rural",
                              estrato == " de 500 a 1 999 habitantes" ~ "Rural",
                              TRUE ~ "Urbano")) 
    
    df <- df |> 
      mutate(p110_n = as.character(p110),
             p111a_n = as.character(p111a),
             p1121_n = as.character(p1121),
             p1142_n = as.character(p1142),
             p1144_n = as.character(p1144)) |> 
      mutate(Agua = case_when(p110_n == "red p\xfablica, fuera de la vivienda pero dentro del edificio" ~ 1,
                              p110_n == "red p\xfablica, dentro de la vivienda" ~ 1,
                              is.na(p110) ~ NA,
                              TRUE ~ 0), 
             Agua_24hrs = case_when(Agua == 1 & p110c1 == 24 ~ 1,
                                    is.na(p110c1) ~ NA,
                                    is.na(p110) ~ NA,
                                    TRUE ~ 0),
             Saneamiento = case_when(p111a_n == "red p\xfablica de desag\xfce fuera de la vivienda pero dentro del edificio" ~ 1,
                                     p111a_n == "red p\xfablica de desag\xfce dentro de la vivienda" ~ 1,
                                     is.na(p111a) ~ NA,
                                     TRUE ~ 0),
             Electricidad = case_when(p1121_n == "electricidad" ~ 1,
                                      is.na(p1121) ~ NA,
                                      TRUE ~ 0),
             Celular = case_when(p1142_n == "celular" ~ 1,
                                 p1142_n == "tel\xe9fono celular" ~ 1,
                                 is.na(p1142) ~ NA,
                                 TRUE ~ 0),
             Internet = case_when(p1144_n == "internet" ~ 1,
                                  p1144_n == "conexi\xf3n a internet" ~ 1,
                                  is.na(p1144) ~ NA,
                                  TRUE ~ 0)) |> 
      mutate(Combo_servicios = case_when(Agua == 1 & Saneamiento == 1 & Electricidad == 1 & Celular == 1 & Internet == 1 ~ 1,
                               is.na(Agua) ~ NA, is.na(Saneamiento) ~ NA, is.na(Electricidad) ~ NA, is.na(Celular) ~ NA, is.na(Internet) ~ NA,
                               TRUE ~ 0))    
  ## Now we define the sample design
    
  enaho_design <- svydesign(id = df$conglome, data = df, strata = df$estrato, weights = ~df$factor07, vars = list(region = ~dpto))
  options(survey.lonely.psu="adjust")
  

  Agua <- svyby(~ Agua, ~ dpto, design = enaho_design, svymean, na.rm = TRUE) |> 
    subset(select = -se)
  Agua_24hrs <- svyby(~ Agua_24hrs, ~ dpto, design = enaho_design, svymean, na.rm = TRUE)|> 
    subset(select = -se)
  Saneamiento <- svyby(~ Saneamiento, ~ dpto, design = enaho_design, svymean, na.rm = TRUE)|> 
    subset(select = -se)
  Electricidad <- svyby(~ Electricidad, ~ dpto, design = enaho_design, svymean, na.rm = TRUE)|> 
    subset(select = -se)
  Celular <- svyby(~ Celular, ~ dpto, design = enaho_design, svymean, na.rm = TRUE)|> 
    subset(select = -se)
  Internet <- svyby(~ Internet, ~ dpto, design = enaho_design, svymean, na.rm = TRUE)|> 
    subset(select = -se)
  Combo_servicios <- svyby(~ Combo_servicios, ~ dpto, design = enaho_design, svymean, na.rm = TRUE)|> 
    subset(select = -se)
  
  df <- reduce(list(Agua, Agua_24hrs, Saneamiento, Electricidad, Celular, Internet, Combo_servicios), merge, by = "dpto")  |> 
    mutate(depto = case_when(dpto == 1 ~ "Amazonas", dpto == 2 ~ "Ancash", dpto == 3 ~ "Apurimac", dpto == 4 ~"Arequipa", dpto ==  5 ~ "Ayacucho", dpto ==  6 ~ "Cajamarca",
                             dpto == 7 ~ "Callao", dpto == 8 ~ "Cusco", dpto == 9 ~ "Huancavelica", dpto == 10 ~ "Hu??nuco", dpto == 11~ "Ica", dpto == 12 ~ "Jun??n",
                             dpto ==13 ~ "La Libertad", dpto ==14 ~ "Lambayeque", dpto == 15 ~ "Lima", dpto == 16 ~ "Loreto", dpto == 17 ~ "Madre de Dios", dpto == 18 ~ "Moquegua",
                             dpto == 19 ~ "Pasco", dpto == 20 ~ "Piura", dpto == 21 ~ "Puno", dpto == 22 ~ "San Mart??n", dpto == 23 ~ "Tacna", dpto == 24 ~ "Tumbes", dpto == 25 ~ "Ucayali")) |> 
    mutate(anio = substr(df_name, nchar(df_name) - 3, nchar(df_name)))
  assign(df_name, df)  
  }
  
  data_shiny <- rbind(data_2017, data_2018, data_2019, data_2020, data_2021) |> 
    mutate(UBIGEO = sprintf("%06d", dpto * 10000)) 
  
  lista<-data_shiny |> 
    select(depto) |> 
    distinct() |> 
    arrange(depto) |> 
    drop_na()
  
  lista_anios<-data_shiny |> 
    select(anio) |> 
    distinct() |> 
    arrange(anio) |> 
    drop_na()
  
  map_peru <- map_DEP |> #Cargamos la base de datos sobre los departamentos del Peru
    rename(UBIGEO = COD_DEPARTAMENTO ) #renombramos la variable del DF para el merge por UBIGEO
  
  map_shiny <- merge(x = map_peru, y = data_shiny, by = "UBIGEO", all.x = TRUE)

  list_indicadores <- colnames(data_shiny)[2:8]
  
  # Generate a palette with 25 gradient colors
  my_palette <- colorRampPalette(c("white", "#26185F"))(25)

  
# 3. UI: USER INTERFACE -----
# UI (User Interface): La UI es la interfaz de usuario y es la primera parte que los usuarios ven 
# cuando utilizan la aplicaci??n. 
# La UI se define utilizando la funci??n ui.R y contiene los componentes gr??ficos, 
# como men??s, botones, gr??ficos y tablas.

ui <- fluidPage(
        navbarPage("Observatorio Regional", theme = shinytheme("lumen"),
                  tabPanel("Mapa interactivo", fluid = TRUE, icon = icon("globe-americas"),
                           sidebarLayout(
                             sidebarPanel(
                               titlePanel("Seleccione el a??o e indicador de inter??s"),
                               selectInput(inputId = "Year",
                                           label = "Seleccione el a??o",
                                           choices = lista_anios,
                                           selected = 2021,
                                           width = "220px"
                               ),
                               selectInput(inputId = "Indicador",
                                           label = "Seleccione el indicador",
                                           choices = list_indicadores,
                                           selected = "Combo_servicios",
                                           width = "220px"
                               )),
                             mainPanel(
                               fluidRow(
                                 column(3, offset = 9)),
                               withSpinner(plotOutput(outputId = "map1"
                               )),
                               hr(),
                               fluidRow(column(7, helpText("Ingrese el nombre del departamento para revisar la informaci??n de los ??ltimos a??os")
                               )),
                               column(width = 2, offset = 2, conditionalPanel(
                                 condition = "output.tabla1",
                                 actionButton(inputId = "FinderClear", label = "Clear Table"))),
                               br(),
                               fluidRow(
                                 withSpinner(dataTableOutput(outputId = "tabla1"))),
                             ))),
                  tabPanel("Sem??foro regional", fluid = TRUE, icon = icon("chart-bar"),
                           sidebarLayout(
                             sidebarPanel(
                               titlePanel("Seleccione el departamento de su inter??s"),
                               selectInput(inputId = "Depto",
                                           label = "Seleccione el departamento",
                                           choices = lista,
                                           selected = "Lima",
                                           width = "220px"
                               )),
                             mainPanel(
                               fluidRow(
                                 column(6, plotOutput("plot1")),
                                 column(6, plotOutput("plot2"))
                                 ),
                               fluidRow(
                                 column(6, plotOutput("plot3")),
                                 column(6, plotOutput("plot4"))
                               ),
                               fluidRow(
                                 column(6, plotOutput("plot5")),
                                 column(6, plotOutput("plot6"))
                               )
                               ))
                           )
                  
                ))



# 4. SERVIDOR ----- 
# El servidor maneja la l??gica de la aplicaci??n, 
# procesa los datos y crea la salida que se muestra en la UI. 

server <- function(input, output) {

  output$map1 <- renderPlot({
    
    map_shiny |> 
      filter(anio == input$Year) |> 
      ggplot() +
      aes(geometry = geometry) +
      geom_sf(aes_string(fill = input$Indicador)) +
      theme(axis.text.x = element_blank(), axis.text.y = element_blank(), axis.ticks = element_blank())
    
  }) 
  
  
  output$tabla1 <- renderDataTable({
    
    data_shiny |> 
      mutate(n_Agua = percent(Agua, accuracy = 0.01), n_Agua_24hrs = percent(Agua_24hrs, accuracy = 0.01), n_Saneamiento = percent(Saneamiento, accuracy = 0.01),
             n_Electricidad = percent(Electricidad, accuracy = 0.01), n_Celular = percent(Celular, accuracy = 0.01), n_Internet = percent(Internet, accuracy = 0.01), n_Combo_servicios = percent(Combo_servicios, accuracy = 0.01)) |> 
      select(dpto, depto, anio, UBIGEO, n_Agua, n_Agua_24hrs, n_Saneamiento, n_Electricidad, n_Celular, n_Internet, n_Combo_servicios) |> 
      rename(Agua = n_Agua, Agua_24hrs=n_Agua_24hrs, Saneamiento=n_Saneamiento, Electricidad=n_Electricidad, Celular=n_Celular, Internet=n_Internet, Combo_servicios=n_Combo_servicios) |> 
      select(anio, depto, input$Indicador)
       
    
  })
  
  output$plot1 <- renderPlot({
    
    data_shiny |> 
      mutate(n_Agua = percent(Agua, accuracy = 0.01), n_Agua_24hrs = percent(Agua_24hrs, accuracy = 0.01), n_Saneamiento = percent(Saneamiento, accuracy = 0.01),
             n_Electricidad = percent(Electricidad, accuracy = 0.01), n_Celular = percent(Celular, accuracy = 0.01), n_Internet = percent(Internet, accuracy = 0.01), n_Combo_servicios = percent(Combo_servicios, accuracy = 0.01)) |> 
      select(dpto, depto, anio, UBIGEO, n_Agua, n_Agua_24hrs, n_Saneamiento, n_Electricidad, n_Celular, n_Internet, n_Combo_servicios) |> 
      rename(Agua = n_Agua, Agua_24hrs=n_Agua_24hrs, Saneamiento=n_Saneamiento, Electricidad=n_Electricidad, Celular=n_Celular, Internet=n_Internet, Combo_servicios=n_Combo_servicios) |> 
      filter(depto==input$Depto) |> 
      ggplot(aes(y=Agua, x=anio)) +
      geom_col() + 
      geom_point() +
      geom_text(aes(label = Agua), vjust = -0.5)+
      labs(x = "A??o", 
           y = "Porcentaje de hogares con acceso a agua", 
           title = "Acceso a agua",
           subtitle = "2017-2021", 
           caption = "Fuente:INEI")+
      theme_classic()
    
  })
  
  output$plot2 <- renderPlot({
    
    data_shiny |> 
      mutate(n_Agua = percent(Agua, accuracy = 0.01), n_Agua_24hrs = percent(Agua_24hrs, accuracy = 0.01), n_Saneamiento = percent(Saneamiento, accuracy = 0.01),
             n_Electricidad = percent(Electricidad, accuracy = 0.01), n_Celular = percent(Celular, accuracy = 0.01), n_Internet = percent(Internet, accuracy = 0.01), n_Combo_servicios = percent(Combo_servicios, accuracy = 0.01)) |> 
      select(dpto, depto, anio, UBIGEO, n_Agua, n_Agua_24hrs, n_Saneamiento, n_Electricidad, n_Celular, n_Internet, n_Combo_servicios) |> 
      rename(Agua = n_Agua, Agua_24hrs=n_Agua_24hrs, Saneamiento=n_Saneamiento, Electricidad=n_Electricidad, Celular=n_Celular, Internet=n_Internet, Combo_servicios=n_Combo_servicios) |> 
      filter(depto==input$Depto) |> 
      ggplot(aes(y=Agua_24hrs, x=anio)) +
      geom_col() + 
      geom_point() +
      geom_text(aes(label = Agua_24hrs), vjust = -0.5)+
      labs(x = "A??o", 
           y = "Porcentaje de hogares con acceso a agua las 24 horas del d??a", 
           title = "Acceso a agua continuo",
           subtitle = "2017-2021", 
           caption = "Fuente:INEI")+
      theme_classic()
    
  })  
  
  output$plot3 <- renderPlot({
    
    data_shiny |> 
      mutate(n_Saneamiento = percent(Saneamiento, accuracy = 0.01)) |> 
      select(dpto, depto, anio, UBIGEO, n_Saneamiento) |> 
      rename(Saneamiento=n_Saneamiento) |> 
      filter(depto==input$Depto) |> 
      ggplot(aes(y=Saneamiento, x=anio)) +
      geom_col() + 
      geom_point() +
      geom_text(aes(label = Saneamiento), vjust = -0.5)+
      labs(x = "A??o", 
           y = "Porcentaje de hogares con acceso a desag??e", 
           title = "Acceso a desag??e",
           subtitle = "2017-2021", 
           caption = "Fuente:INEI")+
      theme_classic()
    
  })
  
  output$plot4 <- renderPlot({
    
    data_shiny |> 
      mutate(n_Electricidad = percent(Electricidad, accuracy = 0.01)) |> 
      select(dpto, depto, anio, UBIGEO, n_Electricidad) |> 
      rename(Electricidad=n_Electricidad) |> 
      filter(depto==input$Depto) |> 
      ggplot(aes(y=Electricidad, x=anio)) +
      geom_col() + 
      geom_point() +
      geom_text(aes(label = Electricidad), vjust = -0.5)+
      labs(x = "A??o", 
           y = "Porcentaje de hogares con acceso a electricidad", 
           title = "Acceso a electricidad",
           subtitle = "2017-2021", 
           caption = "Fuente:INEI")+
      theme_classic()
    
  })  
  
  output$plot5 <- renderPlot({
    
    data_shiny |> 
      mutate(n_Celular = percent(Celular, accuracy = 0.01)) |> 
      select(dpto, depto, anio, UBIGEO, n_Celular) |> 
      rename(Celular=n_Celular) |> 
      filter(depto==input$Depto) |> 
      ggplot(aes(y=Celular, x=anio)) +
      geom_col() + 
      geom_point() +
      geom_text(aes(label = Celular), vjust = -0.5)+
      labs(x = "A??o", 
           y = "Porcentaje de hogares con acceso a tel??fono celular", 
           title = "Acceso a tel??fono celular",
           subtitle = "2017-2021", 
           caption = "Fuente:INEI")+
      theme_classic()
    
  })
  
  output$plot6 <- renderPlot({
    
    data_shiny |> 
      mutate(n_Internet = percent(Internet, accuracy = 0.01)) |> 
      select(dpto, depto, anio, UBIGEO, n_Internet) |> 
      rename(Internet=n_Internet) |> 
      filter(depto==input$Depto) |> 
      ggplot(aes(y=Internet, x=anio)) +
      geom_col() + 
      geom_point() +
      geom_text(aes(label = Internet), vjust = -0.5)+
      labs(x = "A??o", 
           y = "Porcentaje de hogares con acceso a internet", 
           title = "Acceso a internet",
           subtitle = "2017-2021", 
           caption = "Fuente:INEI")+
      theme_classic()
    
  })  
}

# 5. EJECUCI??N DE APLICACI??N ----- 
shinyApp(ui = ui, server = server)
  
  
  