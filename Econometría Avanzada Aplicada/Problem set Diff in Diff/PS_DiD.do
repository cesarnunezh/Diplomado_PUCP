/*******************************************************************************
Autores: Mauricio Ibáñez & César Núñez
Title: Problem Set - Differences in Differences
Date: 30/06/2023
*******************************************************************************/

/*******************************************************************************
Índice: 
1. Problem 1 - Descarga de datos
2. Problem 1 - Respuestas
3. Problem 2 - Descarga de datos
4. Problem 2 - Respuestas
********************************************************************************/

/*******************************************************************************
Problem 1 - Descarga de datos
*******************************************************************************/
{
global bd "D:\1. Documentos\1. Estudios\7. Diplomado PUCP Data Science\06. Econometría Aplicada Avanzada\1. Data\PS - DiD"

cd "$bd"
global year "96 97 98 99 00 01 02 03 04 05 06"
foreach yy of global year {

if `yy'==96 | `yy'==97 | `yy'==98 | `yy'==99 {
shell curl -o cbp`yy'st.zip "https://www2.census.gov/programs-surveys/cbp/datasets/19`yy'/cbp`yy'st.zip"
}
else if `yy'!=96 & `yy'!=97 & `yy'!=98 & `yy'!=99 {
shell curl -o cbp`yy'st.zip "https://www2.census.gov/programs-surveys/cbp/datasets/20`yy'/cbp`yy'st.zip"
}
unzipfile cbp`yy'st.zip, replace
import delimited "cbp`yy'st.txt", clear

if `yy'==96 | `yy'==97 {
rename sic naics
}
else {
}

keep if substr(naics,4,3)=="///"
* keeping all 3-digit manufacturing sector
/*keep if naics=="311///" | naics=="312///" | naics=="313///" | naics=="314///" | naics=="315///" | naics=="316///" | naics=="321///" | naics=="322///" | naics=="323///" | naics=="324///" | naics=="325///" | naics=="326///" | naics=="327///" | naics=="331///" | naics=="332///" | naics=="333//" | naics=="334///" | naics=="335///" | naics=="336///" | naics=="337///" | naics=="339///" */

keep fipstate naics emp qp1 ap est
if `yy'==96 | `yy'==97 | `yy'==98 | `yy'==99 {
	gen year=19`yy' 
}
else if `yy'!=96 | `yy'!=97 | `yy'!=98 | `yy'!=99 {
	gen year=20`yy' 
}
save cbp`yy'st.dta, replace
}

use cbp98st.dta, clear 
append using cbp99st.dta 
append using cbp00st.dta 
append using cbp01st.dta 
append using cbp02st.dta 
append using cbp03st.dta
append using cbp04st.dta 
append using cbp05st.dta
append using cbp06st.dta 

save "$bd/data_problem1.dta", replace
}
/*******************************************************************************
Problem 1 - Respuestas
*******************************************************************************/

use "$bd/data_problem1.dta", clear

* 1. What is the level of observation?

isid fipstate naics year 	/// Each observation correspond to a sector i in the state j at the year t.

* 2. Construct 1 dummy variable called “post_china” where post_china=1 for year>=2001 and 0 otherwise. 

gen post_china = (year>=2001)

tab year post_china  	/* We can verify that the dummy variable was created*/

* 3. Construct 1 dummy variable called “manuf” where manuf=1 for all the observations that start with naics code “3” – which is manufacturing - and 0 otherwise. 

gen fl_naics = substr(naics,1,1)	/// First, we extract the first letter of naics

gen manuf = (fl_naics == "3")		/// Secong, we generate the dummy variable manuf

tab fl_naics manuf	/* We can verify that the dummy variable was created*/

* 4. Construct the values necessary to generate the difference-in-difference estimate (i.e. 2x2 Matrix) of the effect of China entering the WTO on employment (emp). 

* First we calculate the means of every group (T and C) for each year (Pre and Post)
mean emp, over(post_china manuf)

* Now, we store results in a 1x4 matrix 
matrix define did_matrix = e(b)

* Finally, we calculate the DiD estimate by doing the doble difference of means
scalar did_estimate = (did_matrix[1,4] - did_matrix[1,2]) - (did_matrix[1,3] - did_matrix[1,1])

/* The results of the DiD estimate is -3292.5804, which means that the effect of 
China entering to the WTO in 2001 over employment in manufacture was a loss in 
employment of 3292 jobs.*/

* 5. Estimate a diff-in-diff regression and make sure you get the same diff-in-diff estimate as in part 4.

/* Now we estimate the DiD regression and we obtain the same result in part 4. 
But here we can notice that the coefficient is significant.*/
reg emp post_china##manuf , robust


* 6. Estimate a diff-in-diff regression for the effect of China entering the WTO in 2001 on the number of establishment (est), an average pay (qp1/emp). Interpret the results. 

/* We estimate the effect of China entering WTO over the number of establishments
in manufacture sector. The DiD estimate is not significant. */
reg est post_china##manuf , robust

* We generate the variable average payment
gen av_pay = qp1/emp

/* We estimate the effect of China entering WTO over the average payment in manu
facture sector. The DiD estimate is -215 dollars in quarterly average payment.*/
reg av_pay post_china##manuf , robust

* 7. Estimate same regression as in (5) but now take logs of the dependent variable (i,e, log(emp)). Interpret your results. Is it necessary to take logs?

* We generate the variable natural logarithm of employment
gen ln_emp = ln(emp)

/* We estimate the DiD regression of the new variable. The result is that the effect
of China entering WTO over employment is a decrease of 16.7% of the level of employment
 in manufacture sector. */
reg ln_emp post_china##manuf , robust

/* To proof that it is necessary to use logarithm we need to verify the distribu
tion of employment. In order to do that, we plot two histogram graphs of empleyment 
in levels and in logarithm. */
histogram emp if manuf==1, percent fcolor(yellow) legend(label (1 "Manufacture")) addplot(histogram emp if manuf != 1, percent fcolor(blue) ///
legend(label (2 "Others"))) saving(hist1, replace)
  
histogram ln_emp if manuf==1, percent fcolor(yellow) legend(label (1 "Manufacture")) addplot(histogram ln_emp  if manuf != 1, percent fcolor(blue) ///
legend(label (2 "Others"))) saving(hist2, replace)

/*In this graph we can notice that the employment in levels is not normally distri
buted*; but the log employment is more near to be a normal distribution. */
graph combine hist1.gph hist2.gph


* 8. Generate one dummy per year. Construct the interaction between each year dummies and your treatment group (manuf). You should have 9 interaction terms. 

egen dummy_year = group(year)
tabulate year, generate(dummy_)

foreach year of numlist 1/9 {
gen dummy_`year'_manuf = dummy_`year' * manuf 
}


* 9. 

reghdfe ln_emp dummy_2-dummy_9 dummy_2_manuf-dummy_9_manuf, absorb(naics fipstate)

coefplot, keep(dummy_2_manuf dummy_3_manuf dummy_4_manuf dummy_5_manuf dummy_6_manuf dummy_7_manuf dummy_8_manuf dummy_9_manuf)                     ///
          coeflabels(dummy_2_manuf = "1999"													///
                     dummy_3_manuf = "2000"													///
                     dummy_4_manuf  = "2001"												///
                     dummy_5_manuf  = "2002"												///
                     dummy_6_manuf  = "2003"												///
                     dummy_7_manuf  = "2004"                       							///
                     dummy_8_manuf  = "2005"												///
					 dummy_9_manuf  = "2006")                       						///
          vertical                                                                          ///
          yline(0)                                                                          ///
          ytitle("Log points")                                                              ///
          xtitle("Year") ///
          addplot(line @b @at)                                                              ///
          ciopts(recast(rcap))                                                              ///
          rescale(100)                                                                      ///
          scheme(s1mono) 
		  
graph export "$bd/Coefplot.png", replace

/*As shown in the graph, the effects of China entering the WTO are persistent even
 after 5 years of the beggining of the effect. But we have to notice that the effect 
 also was significant at the year 2000, so it may be the case that even one year
 before the entering to WTO this had a effect previous to be done.*/

