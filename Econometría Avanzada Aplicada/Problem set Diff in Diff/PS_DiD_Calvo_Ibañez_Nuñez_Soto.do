/*******************************************************************************
Autores: Gabriela Calvo, Mauricio Ibáñez, César Núñez & Rodrigo Soto
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
* Insertar aquí la dirección donde se almacena la data
global bd "D:\1. Documentos\1. Estudios\7. Diplomado PUCP Data Science\06. Econometría Aplicada Avanzada\1. Data\PS - DiD"

/*******************************************************************************
Definiendo la estructura del log file del Problem 1
*******************************************************************************/
{
global logfile_name "PS_DiD_Calvo_Ibañez_Nuñez_Soto_P1"

	cap log close
	local td: di %td_CY-N-D  date("$S_DATE", "DMY") 
	local td = trim("`td'")
	local td = subinstr("`td'"," ","_",.)
	local td = subinstr("`td'",":","",.)
	log using "${logfile_name}-`td'_1", text replace 
	local today "`c(current_time)'"
	local curdir "`c(pwd)'"
	local newn = c(N) + 1

}
/*******************************************************************************
Problem 1 - Descarga de datos
*******************************************************************************/
{
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
{
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


* 9. Estimate an event study

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
		  
graph export "$bd/Coefplot_1.png", replace

/*As shown in the graph, the effects of China entering the WTO are persistent even
 after 5 years of the beggining of the effect. But we have to notice that the effect 
 also was significant at the year 2000, so it may be the case that even one year
 before the entering to WTO this had a effect previous to be done.*/

* 10. Estimate a similar event study on the log(est) and average pay. Interpret your results
 
* Generate the logarithm of establishment variable and event study for the log of number of establishments 
 
 gen ln_est = ln(est)
 
reghdfe ln_est dummy_2-dummy_9 dummy_2_manuf-dummy_9_manuf, absorb(naics fipstate)

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
		  
graph export "$bd/Coefplot_2.png", replace
 
/*As shown in the graph, the effects of China entering the WTO over the quantity 
of stablishments are negative, significant and persistent. The quantity of establishments 
reduced between 10% and 15% because the China's entry to the WTO. Also, we can notice that
the effects are bigger for 2005 and 2006. And we can notice that there are
no significant effects for the previous period (1998-2000).*/ 

* Generate the logarithm of average payment and event study for the log of average payment

gen ln_av_pay = ln(av_pay)

reghdfe ln_av_pay dummy_2-dummy_9 dummy_2_manuf-dummy_9_manuf, absorb(naics fipstate)

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
		  
graph export "$bd/Coefplot_3.png", replace

/*As shown in the graph, the effect of China entering the WTO over the average 
payment is negative and significant in 2001, 2002, 2005 and 2006; but not significant
for 2003 and 2004. */

log close
}
/*******************************************************************************
Definiendo la estructura del log file del Problem 2
*******************************************************************************/
{
global logfile_name "PS_DiD_Calvo_Ibañez_Nuñez_Soto_P2"

	cap log close
	local td: di %td_CY-N-D  date("$S_DATE", "DMY") 
	local td = trim("`td'")
	local td = subinstr("`td'"," ","_",.)
	local td = subinstr("`td'",":","",.)
	log using "${logfile_name}-`td'_1", text replace 
	local today "`c(current_time)'"
	local curdir "`c(pwd)'"
	local newn = c(N) + 1
}
/*******************************************************************************
Problem 2 - Respuestas
*******************************************************************************/
{
use "$bd\eitc.dta", clear
* 1. Create a table summarizing all the data provided in the data set.
des
sum

* 2. Calculate the sample means of all variables for (a) single women with no children, (b) single women with 1 child, and (c) single women with 2+ children
summarize if children==0
summarize if children == 1
summarize if children >=2

* 3. Construct a variable for the “treatment” called anykids (indicator for 1 or more kids) and a variable for time being after the expansion (called post93—should be 1 for 1994 and later)
gen anykids = (children >= 1)
gen post93 = (year >= 1994)

* 4. Using the “interaction term” diff-in-diff specification, run a regression to estimate the difference-in-differences estimate of the effect of the EITC program on earnings. Use all women with children as the treatment group.
gen inter = post93*anykids
reg work post93 anykids inter, robust

* 5. Repeat (iv), but now include state and year fixed effects [Hint: state fixed effects, are included when we include a dummy variable for each state]. Do you get similar estimated treatment effects compared to (iv)?

egen dummy_state = group(state)
tabulate state, generate(dummy_)

reg work post93 anykids inter dummy_2-dummy_51 , robust

/*The result is very similar to (iv) 
iv 	= 0.0468731 and significative
v 	= 0.0457252 and significative */

* 6. Using the specification from (v), re-estimate this model including urate nonwhite age ed unearn, as well as state and year FEs as controls. Do you get similar estimated treatment effects compared to (v)?

egen dummy_year = group(year)
tabulate year, generate(year_)

reg work post93 anykids inter urate nonwhite age ed unearn dummy_2-dummy_51 year_2-year_6, robust

/*The result is very similar to (iv) and (v) but it is bigger 
iv 	= 0.0468731 and significative
v 	= 0.0457252 and significative
vi 	= 0.0525387 and significative
 */

* 7. Estimate a version of your model that allows the treatment effect to vary by those with 1 or 2+ children. Include all other variables as in (vi). Does the intervention seem to be more effective for one of these groups over the other? Why might this be the case in the real world?

gen treat_1 = (children ==1)
gen treat_2 = (children >= 2)

reg work post93##treat_1 post93##treat_2 urate nonwhite age ed unearn dummy_2-dummy_51 year_2-year_6, robust

/*The results for those with 2+ children (0.0606665) are bigger than for thoses with only 1 child (0.0382126). */

* 8. Estimate a “placebo” treatment model as follows: Take data from only the pre-reform period (up to and including 1993). Drop the rest, or restrict your model to run only if year <= 1993. Estimate the effect for all affected women together, just as in (vi). Introduce a placebo policy that begins in 1992 (so 1992 and 1993 are both “treated” with this fake policy). What do you find? Are your results “reassuring”?

gen placebo = (year == 1992 | year == 1993)


reg work placebo##anykids urate nonwhite age ed unearn dummy_2-dummy_51 year_2 if year <=1993, robust

/* The placebo treatment doesn't appear to affect the employment before the reform period of 1993 */

log close
}