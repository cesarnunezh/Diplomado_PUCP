/********************************************
Práctica Calificada - Instrumental Variables
Autores: MI, GC, CN, RS

Estructura:
	1. Rutas
	
	
**********************************************/

* Rutas 
	
*	global bd0 "C:\Users\User\Dropbox\Otros\Cursos\Diplomado ciencias de datos\7. Econometria\IV"
	global bd0 "D:\1. Documentos\1. Estudios\7. Diplomado PUCP Data Science\06. Econometría Aplicada Avanzada\1. Data\PS - IV" /// Ingrese aquí la dirección donde está almacenado la data.
	
* 2. OLS	
	use "$bd0\card", clear
	
	reg lwage educ c.exper##c.exper age black south smsa smsa66 reg661-reg668 
	reg lwage educ c.exper##c.exper black south smsa smsa66 reg661-reg668 
	
* 4. Primera etapa
	
	ivreg2 lwage c.exper##c.exper black south smsa smsa66 reg661-reg668 (educ=nearc4), first
	
* 5. IV estimation
	
	ivreg2 lwage c.exper##c.exper black south smsa smsa66 reg661-reg668 (educ=nearc4)
	
* 7. 2SLS estimation

	ivreg2 lwage c.exper##c.exper black south smsa smsa66 reg661-reg668 (educ=nearc4 nearc2) , first 
	
	ivreg2 lwage c.exper##c.exper black south smsa smsa66 reg661-reg668 (educ=nearc2)
	est store reg1
	estadd local nearc2 "nearc2"
	
	ivreg2 lwage c.exper##c.exper black south smsa smsa66 reg661-reg668 (educ=nearc4)
	est store reg2
	estadd local nearc2 "nearc4"

	ivreg2 lwage c.exper##c.exper black south smsa smsa66 reg661-reg668 (educ=nearc4 nearc2) 
	est store reg3
	estadd local nearc2 "ambos"
	
	esttab reg1 reg2 reg3, se title("Estimaciones según instrumentos utilizados") stats(N r2 nearc2, label("Observations" "R^2" "Instrumento") fmt( 0 3 )) keep(educ exper c.exper*)
	
* 8. GMM con heterocedasticidad

	ivreg2 lwage c.exper##c.exper black south smsa smsa66 reg661-reg668 (educ=nearc4 nearc2) , robust 
	est store reg4

	ivreg2 lwage c.exper##c.exper black south smsa smsa66 reg661-reg668 (educ=nearc4 nearc2) , robust gmm2s
	est store reg5
	
	esttab reg4 reg5, se title("Estimaciones según método utilizado") stats(N r2 nearc2, label("Observations" "R^2" "Instrumento") fmt( 0 3 )) keep(educ exper c.exper*) mtitles("2SLS" "GMM")
