/*******************************************************************************
Proyecto: Tarea 1 - Big Data 
Autor: MI - CN
Objetivo: Responder las preguntas de la Tarea 1	

Estructura:
	
	0. Direcciones
	1. Parte I
	2. Parte I


********************************************************************************/
* 	0. Direcciones

	clear all
	global bd0 "C:\Users\User\Documents\GitHub\Diplomado_PUCP\Big Data\Data"
	global bd1 "C:\Users\User\Documents\GitHub\Diplomado_PUCP_trabajos\Big Data\Data"

********************************************************************************/
*	1. Parte I

/*	1. Indique la veracidad o falsedad de las siguientes afirmaciones:
		a.	Falso. La media muestral no siempre es la misma en tanto la aleatoriedad
			no te garantiza que siempre tendrás las mismas observaciones y por ende 
			tendrás siempre valores diferentes entre una y otra muestra. 
		b.	Falso, si disminuye el tamaño de muestra el error estandar de la media 
			muestral aumenta. Esto asociado a la Ley de Grandes Números, la cual dice 
			que en la medida que aumente el tamaño de la muestra (hasta el infinito), 
			la media muestral se aproximará cada vez más a la poblacional. En otras 
			palabras, el error estandar disminuye hasta aproximadamente cero cuando 
			aumenta el tamaño de la muestra. 
		c.	Es posible, pero NO es lo más probable. Hay que diferenciar la distribución
			del estimador del estimado. Es decir, puede que se dé el caso en el que, con 
			una muestra pequeña, se obtenga una media muestral muy cercana a la media 
			poblacional (estimado); sin embargo, esto no necesariamente ocurrirá en todos
			los casos si se repite n veces la realización de la muestra del mismo tamaño 
			(distribución del estimador). En conclusión, puede que una media muestral sea
			más cercana a la poblacional incluso con menor tamaño de muestra; sin embargo,
			si este ejercicio se realiza varias veces, lo más probable, es que la que más
			se aproxime en promedio sea la de mayor tamaño de muestra. 
		d.	Verdadero. Sí existe un trade-off entre número de conglomerados y el número
			de observaciones por conglomerados, dado un mismo tamaño de muestra. A mayor 
			número de conglomerados, hay una mayor varianza que puede aportar a las 
			estimaciones, pero esto implica un menor número de varianza dentro de cada 
			cluster lo que afecta también a la precisión de las estimaciones. 
		
	2.	Indique a qué enfoque (diseño o modelo) corresponde cada una de las siguientes características:
		a.	Asume una población finita --> Enfoque de diseño
		b.	Asume que la variable objetivo es aleatoria --> Enfoque de modelo
		c.	Asume un proceso generador de datos --> Enfoque de modelo
		d.	Asume que el componente aleatorio viene de cómo seleccionamos las observaciones --> Enfoque de diseño


	3.	Responda tres de las siguientes preguntas. La respuesta a cada pregunta no debe pasar de 75 palabras.
		
		a. Los estratos son subconjuntos que son heterogéneos entre sí pero homogéneos entre ellos, la idea es extraer una muestra independiente de cada estrato. En cambio, los dominios son sub grupos de la población pre definidios y se busca obtener resultados sobre ellos con cierto margen de error. La muestra agregada de los dominios representan a la población total y suelen ser espacios geográficos o político administrativo
		b. El factor de expansión permite extrapolar los resultados de la variable de interés para el universo de estudio sobre el cual se basó la muestra. Se construye mediante la inversa de la porbabilidad de ser elegido una unidad de muestreo
		c. Se esperaría que el error estándar disminuya dado que al estratificar el diseño muestral obtenemos estimaciones más precisas con la segmentación en grupos.
		

********************************************************************************/
*	2. Parte II
* 	1. Pregunta 1 - Calculo de variables

	use "$bd0/Data para Tarea1.dta", clear

*	1.1. Una variable categórica que identifique los siguientes grupos etarios: de 12 a 24 inclusive, de 25 a 64 inclusive, de 65 a más. Llamar a esta variable “rango_edad”.

	recode edad (12/24 = 1) (25/64 = 2) (nonmissing = 3) , gen(rango_edad)
	
*	1.2. Una variable dicotómica que identifique a las Unidades Agropecuarias (UAs) que son dirigidas por personas cuya lengua materna es una lengua nativa*. Llamar a esta variable “pp_lenguanativa”.

	gen pp_lenguanativa = (lengua <=4)
	gen N = _N
*	2. Establecer una semilla (“seed”) usando los dígitos del código PUCP de alguno de los miembros de la pareja.

	set seed 1100839
	
*	3.	Extraer una Muestra Aleatoria Simple (MAS) de 2247 observaciones de la población total (base completa). Con esta muestra, utilizar la variable “sup_total” y obtener los siguientes estadísticos.

	sample 2247 , count
	
	*3.1.	La suma de todos los valores
	egen suma_total = total(sup_total)
	
	*3.2.	La media
	egen media = mean(sup_total)
	
	*3.3.	La fracción muestral
	gen frac_muestral = 2247/N
	
	*3.4.	La corrección de población finita (fpc)
	gen corr_pob_finita = 1 - frac_muestral
	
	*3.5.	La suma de desviaciones al cuadrado
	gen desv_2 = (sup_total - media)^2
	egen sum_desv_2 = total(desv_2)

	*3.6.	La varianza de la muestra
	gen var_muestra = (1/2246)*sum_desv_2
	
	*3.7.	La varianza muestral de la media
	gen var_muestral_media = var_muestra * corr_pob_finita / 2247
	
	*3.8.	El error estándar de la media
	gen se_media = var_muestral_media^0.5
	
	*3.9.	El factor de expansión de cada observación
	gen factor_mas = N/2247 /// La sumatoria corresponde a la población total. Porque cada observación de la muestra tiene el mismo 
	
*	4.	Extraer una Muestra Aleatoria Estratificada (MAE) de 2247 observaciones utilizando la variable “rango_edad” como estrato. Con esta muestra, utilizar la variable “pp_lenguanativa” y: 

	use "$bd1/Data para Tarea1.dta", clear

	recode edad (12/24 = 1) (25/64 = 2) (nonmissing = 3) , gen(rango_edad)
	
	gen pp_lenguanativa = (lengua <=4)
	gen N = _N
	
	egen estrato = group(rango_edad)
	sort estrato
	
	set seed 1100839

	bys estrato: sample 0.1
	bys estrato: count
	
	gen pw =  114647/115 if estrato == 1
	replace pw = 1665605/1666 if estrato == 2
	replace pw = 466450/466 if estrato == 3
	
	gen fpc = 114647 if estrato == 1
	replace fpc = 1665605 if estrato == 2
	replace fpc = 466450 if estrato == 3
	
	*bsample round(0.001*_N), strata(estrato) 
	
*	4.1.	Obtener los mismos estadísticos de las preguntas 3.1-3.9 según la variable de estratificación (intra estrato).

	*1.	La suma de todos los valores
	bys rango_edad: egen suma_total = total(pp_lenguanativa)
	
	*2.	La media
	bys rango_edad: egen media = mean(pp_lenguanativa)
	
	*3.	La fracción muestral
	gen frac_muestral=  115/114647 if estrato == 1
	replace frac_muestral = 1666 /1665605 if estrato == 2
	replace frac_muestral = 466/466450 if estrato == 3
	
	*4.	La corrección de población finita (fpc)
	gen corr_pob_finita = 1 - frac_muestral if estrato == 1
	replace  corr_pob_finita = 1 - frac_muestral if estrato == 2
	replace  corr_pob_finita = 1 - frac_muestral if estrato == 3
	
	*5.	La suma de desviaciones al cuadrado
	bys rango_edad: gen desv_2 = (pp_lenguanativa - media)^2
	bys rango_edad: egen sum_desv_2 = total(desv_2)

	*6.	La varianza de la muestra
	gen var_muestra = (1/114)*sum_desv_2 if estrato == 1
	replace var_muestra = (1/1665)*sum_desv_2 if estrato == 2
	replace var_muestra = (1/465)*sum_desv_2 if estrato == 3	
	
	*7.	La varianza muestral de la media
	gen var_muestral_media = var_muestra * corr_pob_finita / 115 if estrato == 1
	replace var_muestral_media = var_muestra * corr_pob_finita / 1666 if estrato == 2
	replace var_muestral_media = var_muestra * corr_pob_finita / 466 if estrato == 3
	
	*8.	El error estándar de la media
	bys rango_edad: gen se_media = var_muestral_media^0.5
	
	*9.	El factor de expansión de cada observación
	gen factor_exp = (1/frac_muestral)

*	4.2.	Declare el diseño muestral y obtenga el promedio de la variable “pp_lenguanativa” para toda la muestra utilizando el diseño muestral

	svyset, clear 
	svyset [pweight = pw], strata(estrato) fpc(fpc)
	svydes
	
	svy: mean pp_lenguanativa

*	4.3.	Discuta como pueden utilizarse los resultados obtenidos en 4.1 para poder calcular el promedio de la variable “pp_lenguanativa” para toda la muestra.

	* Se puede calcular el promedio de la muestra para la variable "pp_lenguanativa" mediante la división entre i) la suma de las medias de cada strato ponderado por el tamaño de cada estrato y ii) el tamaño de la muestra. De esta manera:
	
	tab media
	display ((0.2956522*115) + (0.3829532*1666) + (0.3969957*466))/2247