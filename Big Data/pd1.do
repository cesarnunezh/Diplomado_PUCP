
/*==============================================================================

Práctica dirigida 1

==============================================================================*/

clear all

*** Rutas ***
global input "G:\Mi unidad\QLAB\Sampling\PD1"


*---------------------------------------*
*** Algunos experimentos con muestras ***
*---------------------------------------*
	
	* Abrir base de datos y generar indicadores *
	use "$input\cpv2017_pob07mod" , clear
	gen mujer = c5_p2 == 2
	rename c5_p4_1 edad
	gen trabaja = c5_p16 == 1 | inlist(c5_p17 , 1,2,3,4,5)
	replace trabaja = . if edad < 5
	
	h sample // Revisar comando sample
	
	* Muestras "pequeñas" *
	cls
	foreach x of numlist 0.1(0.1)2 {
	preserve
		dis "" 
		dis "Muestra de `x'%"
		sample `x'
		mean mujer edad
	restore
	}
	
	mean mujer edad
	
	* Muestras grandes *
	cls
	foreach x of numlist 2(1)10 {
	preserve
		dis "" 
		dis "Muestra de `x'%"
		sample `x'
		mean mujer edad
	restore
	}
	
	mean mujer edad
	
	
	* Muestras repetidas del mismo tamaño *
	cls
	foreach x in 0.01 0.01 0.01 0.01 0.01 {
	preserve
		dis "" 
		dis "Muestra de `x'%"
		sample `x'
		mean mujer edad
	restore
	}
	
	cls
	foreach x in 1 1 1 1 1 {
	preserve
		dis "" 
		dis "Muestra de `x'%"
		sample `x'
		mean mujer edad
	restore
	}
	

*-----------------------*
*** Cálculo del poder ***
*-----------------------*
	set scheme s1color
	h power // revisar comando power
	summ trabaja
	
	* Tamaño de la muestra *
	power onemean 0.49 0.5
	power onemean 0.49 0.5 , sd(0.5)
	
	power onemean 0.49 (0.5(0.01)0.59) , graph
	power onemean 0.49 (0.5(0.01)0.59) , sd(0.5) graph
	power onemean 0.49 (0.5(0.01)0.59) , sd(0.5 1 1.5 2 2.5) graph
	
	power oneproportion 0.49 (0.5(0.01)0.59) , graph
	power oneproportion 0.1 (0.12(0.02)0.2) , graph
	
	* Poder en base a una muestra *
	power onemean 0.49 0.5 , n(78491)
	power onemean 0.49 (0.5(0.01)0.59)  , n(78491) graph
	power onemean 0.49 (0.5(0.01)0.59)  , n(200(200)1000) graph
	power onemean 0.49 (0.5(0.01)0.59)  , n(500(500)2500) graph
	
	
	* Para evaluaciones de impacto usar opciones twomeans *
	* Tamaño de la muestra *
	power twomeans 1 2
	power twomeans 1 (2(1)5) , graph
	power twomeans 1 (2(1)5) , sd(0.5 1 1.5 2 2.5) graph
	power twomeans 1 2 , sd1(0.5 1 1.5 2 2.5) sd2(0.5 1 1.5 2 2.5) graph
	power twomeans 100 110 , sd(100) // Ojo con la desviación estándar
	
	
	
	
	
	
	
	
	
	
	


