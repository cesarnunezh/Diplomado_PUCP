********************************************************************************
* 	   QLAB - CURSO BIG DATA & ANALYTICS PARA LA GESTIÓN PÚBLICA PERUANA	   *
*      							TEMA: MAS Y MAE								   *
*							Fecha: agosto del 2023							   *
********************************************************************************

/*______________________________________________________________________________________________
**Obejtivos del do file:_
  *Repasar cómo utilizar el comando "sample" para extraer MAS (muestras aleatorias simples)
  *Aprender a utilizar el comando "sample" para extraer MAE (muestras aleatorias estratificadas)
  *Aprender a utilizar el comando "svyset" para anunciar el diseño muestral (MAS o MAE)
  *Aprender a utilizar el prefijo "svy:" para calcular estadísticos considerando el diseño muestral
______________________________________________________________________________________________*/


clear all

* Establecemos las carpetas de trabajo:

global main "G:\Mi unidad\QLAB\Sampling\PD2"


*-----------------*
*** Ejercicio 1 ***
*-----------------*

	* CPV *
	use "$main\cpv2017_pob07mod" , clear
	gen mujer = c5_p2 == 2
	rename c5_p4_1 edad
	gen trabaja = c5_p16 == 1 | inlist(c5_p17 , 1,2,3,4,5)
	replace trabaja = . if edad < 5
	set seed 12345

	
	* MAS *
	preserve
		set seed 12345 
		sample 1 // Tomar la muestra
		mean mujer edad
		mean edad , over(mujer)
	restore
	
	* MAE *
	preserve
		set seed 12345
		sample 1 , by(mujer) 			// Tomar muestra estratificada por sexo
		svyset , strata(mujer) 			// Anunciar el diseño muestral con el comando svyset
		
		svy: mean mujer edad // Calcular estadísticos CONSIDERANDO el diseño muestral (usando de prefijo "svy")
		mean mujer edad		 // Calcular estadísticos SIN CONSIDERAR el diseño muestral
		svy: mean edad , over(mujer)
		mean edad , over(mujer)
	restore
	
	* Pregunta: qué sucede con los SE?
	
	
	* MAE (ejemplo con fpc & pw) *
	set seed 54321
	preserve
		tab mujer
		sample 1 , by(mujer) // Tomar muestra estratificada por sexo
		gen maefpc = 508712 if mujer == 1 // Finite Population Correction --> Cómo lo definimos?
			replace maefpc = 485782 if mujer == 0
		gen maepw = 508712 / 5087 if mujer == 1 // Probability weights --> Cómo lo definimos?
			replace maepw = 485782 / 4858
		svyset [pweight = maepw] , strata(mujer) fpc(maefpc)
		
		svy: mean mujer edad
		mean mujer edad
		svy: mean edad , over(mujer)
		mean edad , over(mujer)
	restore
	

	
	
	
	
	
	
	

