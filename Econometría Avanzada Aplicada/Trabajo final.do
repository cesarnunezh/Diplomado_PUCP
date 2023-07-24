/*******************************************************************************
Proyecto: Trabajo Final - Econometría Aplicada Avanzada
Autor: MI - CN
Objetivo: Estimar estadísticas compras hasta 8UIT	

Estructura:
	
	1. Direcciones
	2. Universo 2022
 	3. Compras 8UIT 2018-2022
 	4. Sanciones e Inhabilitaciones
 	5. Armado de base final del universo

********************************************************************************/
* 	1. Direcciones

	clear all
	global bd0 "D:\1. Documentos\0. Bases de datos\10. OSCE\0. Originales"
	global bd1 "D:\1. Documentos\0. Bases de datos\10. OSCE\1. Bases"
	global bd2 "D:\1. Documentos\0. Bases de datos\10. OSCE\2. Resultados"

********************************************************************************
* 	2. Adjudicaciones mayores a 8 UIT	
{
	foreach x of numlist 2019 2020 2021 2022 {

	import excel "$bd0/CONOSCE_CONVOCATORIAS`x'_0.xlsx", sheet("CONOSCE") firstrow case(lower) clear

	rename (a-ab) (CODIGOENTIDAD	ENTIDAD_RUC	ENTIDAD	TIPOENTIDAD	CODIGOCONVOCATORIA	DESCRIPCION_PROCESO	PROCESO	TIPO_COMPRA	OBJETOCONTRACTUAL	SECTOR	SISTEMA_CONTRATACION	TIPOPROCESOSELECCION	MONTOREFERENCIAL	N_ITEM	DESCRIPCION_ITEM	UNIDAD_MEDIDA	ESTADOITEM	PAQUETE	CODIGOITEM	ITEMCUBSO	DISTRITO_ITEM	PROVINCIA_ITEM	DEPARTAMENTO_ITEM	MONTO_REFERENCIAL_ITEM	MONEDA	FECHA_CONVOCATORIA	FECHAINTEGRACIONBASES	FECHAPRESENTACIONPROPUESTA)
	rename *, lower 
	drop in 1

	save "$bd1/convocatoria_`x'.dta", replace

	import excel "$bd0/CONOSCE_ADJUDICACIONES`x'_0.xlsx", sheet("CONOSCE") firstrow case(lower) clear

	rename (a-r) (CODIGOENTIDAD	ENTIDAD_RUC	CODIGOCONVOCATORIA	PROCESO	N_ITEM	DESCRIPCION_ITEM	ESTADO_ITEM	CANTIDAD_ADJUDICADO_ITEM	MONTO_REFERENCIAL_ITEM	MONTO_ADJUDICADO_ITEM	MONEDA	UNIDAD_MEDIDA	RUC_PROVEEDOR	PROVEEDOR	TIPO_PROVEEDOR	FECHA_CONVOCATORIA	FECHA_BUENAPRO	FECHA_CONSENTIMIENTO_BP)
	rename *, lower 
	drop in 1

	merge m:1 codigoconvocatoria n_item using "$bd1/convocatoria_`x'.dta", keepusing(objetocontractual)

	save "$bd1/adjudicaciones_`x'.dta", replace

	}
}
********************************************************************************
*	3. Ordenes de compras
{
	global meses "JULIO AGOSTO SETIEMBRE OCTUBRE NOVIEMBRE DICIEMBRE"

	foreach x of numlist 2019 2020 2021 2022 {

	foreach y in $meses { 

	import excel "$bd0/CONOSCE_ORDENESCOMPRA`y'`x'_0.xlsx", sheet("CONOSCE") firstrow case(lower) clear

	rename (a-r) (ENTIDAD	RUC_ENTIDAD	FECHA_REGISTRO	FECHA_DE_EMISION	FECHA_COMPROMISO_PRESUPUESTAL	FECHA_DE_NOTIFICACION	TIPOORDEN	NRO_DE_ORDEN	ORDEN	DESCRIPCION_ORDEN	MONEDA	MONTO_TOTAL_ORDEN_ORIGINAL	OBJETOCONTRACTUAL	ESTADOCONTRATACION	TIPODECONTRATACION	DEPARTAMENTO__ENTIDAD	RUC_CONTRATISTA	NOMBRE_RAZON_CONTRATISTA)

	rename *, lower
	drop in 1

	save "$bd1/ordenes_`y'_`x'.dta" , replace
	}
	}


	global meses "AGOSTO SETIEMBRE OCTUBRE NOVIEMBRE DICIEMBRE"

	foreach x of numlist 2019 2020 2021 2022 {

		use "$bd1/ordenes_JULIO_`x'.dta", clear

	foreach y in $meses { 


		append using "$bd1/ordenes_`y'_`x'.dta"
		
		
		
	}
		duplicates drop
		destring monto_total_orden_original, gen(monto_orden)
		
		gen numeric_date = date(fecha_de_emision, "YMD")

		keep if numeric_date > date("`x'-06-30", "YMD")
		
		keep if tipodecontratacion == "Contrataciones hasta 9 UIT (D.U. 016 - 2022) (No incluye las derivadas de contrataciones por catálogo electrónico.)" | tipodecontratacion == "Contrataciones hasta 8 UIT (LEY 30225)(No incluye las derivadas de contrataciones por catálogo electrónico.)"

		histogram monto_orden , by(objetocontractual)
		
		 
		save "$bd1/ordenes_`x'.dta", replace
	}
}
********************************************************************************
* 	4. Unión de bases mayores y menoresa 8 UIT  
{
	use "$bd1/ordenes_2019.dta", clear

	append using "$bd1/ordenes_2020.dta"
	append using "$bd1/ordenes_2021.dta"
	append using "$bd1/ordenes_2022.dta"
	
	gen anio = year(numeric_date)
	
	gen uit = 4200 if anio==2019
	replace uit = 4300 if anio == 2020
	replace uit = 4400 if anio == 2021
	replace uit = 4600 if anio == 2022
	
	gen monto_uit = monto_orden / uit
	
	save "$bd1/ordenes_2019_2022.dta", replace
	
	
	use "$bd1/adjudicaciones_2019.dta", clear
	
	append using "$bd1/adjudicaciones_2020.dta"
	append using "$bd1/adjudicaciones_2021.dta"
	append using "$bd1/adjudicaciones_2022.dta"
	
	gen numeric_date = date(fecha_convocatoria, "DMY")
	gen fecha_adj = date(fecha_buenapro, "DMY")
	
	gen mes_convoc = month(numeric_date)
	gen mes_adj = month(fecha_adj)
	keep if mes_adj >=7
	
	gen anio = year(fecha_adj)
	
	gen uit = 4200 if anio==2019
	replace uit = 4300 if anio == 2020
	replace uit = 4400 if anio == 2021
	replace uit = 4600 if anio == 2022

	destring monto_adjudicado_item, gen(monto_adj)
	
	gen monto_uit = monto_adj / uit
	
	keep if objetocontractual != "Obra"
	
	replace objetocontractual = "BIENES" if objetocontractual =="Bien"
	replace objetocontractual = "CONSULTORIAS OBRAS" if objetocontractual =="Consultoría de Obra"
	replace objetocontractual = "SERVICIOS" if objetocontractual =="Servicio"
	
	rename (ruc_proveedor entidad_ruc) (ruc_contratista ruc_entidad)
	save "$bd1/adjudicaciones_2019_2022.dta", replace

	use "$bd1/ordenes_2019_2022.dta", clear
	
	append using "$bd1/adjudicaciones_2019_2022.dta"	
	keep objetocontractual anio monto_uit ruc_contratista moneda ruc_entidad numeric_date

	gen dummy_2022 = (anio==2022)
	
	save "$bd1/bdfinal.dta" , replace
	}	
	
*	5. Sanciones e inhabilitaciones
{	
	* Penalidades
	import delimited "$bd0\penalidades2018.csv", varnames(1) clear
	drop if objetocontrato!="BIENES"
	tempfile penalidades2018
	save `penalidades2018'
	
	import delimited "$bd0\penalidades2021.csv", varnames(1) clear
	drop if objetocontrato!="Bien"
	replace tipopenalidad=strupper(tipopenalidad)
	replace objetocontrato=strupper(objetocontrato)
	drop if objetocontrato!="BIEN"
	destring idcontrato, replace
	save "$bd1\penalidades2018-2022", replace
	
	append using "`penalidades2018'"
	
	replace monto=subinstr(monto,";","",.)
	replace monto=subinstr(monto,",",".",1)
	replace objetocontrato="BIENES"
	
	destring monto, replace
	
	gen fecha_penalidad = date(fechapenalidad,"DMY",2050)
	gen año = year(fecha_penalidad)
	keep if año>2016
	
	preserve
	collapse (sum) monto_penalidades_acum=monto (count) n_penalidades_acum=monto , by(ruccontratista) 
	tempfile datos_acumulados
	save `datos_acumulados'
	restore
	
	collapse (sum) monto (count) n_penalidades=monto , by(año ruccontratista) 
	collapse (mean) monto_penalidades_prom=monto n_penalidades_prom=n_penalidades, by(ruccontratista)
	merge 1:1 ruccontratista using "`datos_acumulados'"
	drop _m
	
	rename ruccontratista ruc_contratista
	label var monto_penalidades_prom "Monto promedio de las penalidades entre 2018 y 2022"
	label var n_penalidades_prom "Número promedio anual de penalidades entre 2018 y 2022"
	label var monto_penalidades_acum "Monto acumulado de las penalidades entre 2018 y 2022"
	label var n_penalidades_acum "Número acumulado de las penalidades entre 2018 y 2022"
	save "$bd1/contratistas_penalidades" , replace
	
	* Sanciones inhabilitaciones
	
	import delimited "$bd0\sancionados_inhabilitacion.csv", varnames(1) clear

	tostring fecha_inicio, replace
	tostring fecha_fin, replace
	
	gen año_inicio = substr(fecha_inicio,1,4)
	destring año_inicio , replace
	gen mes_inicio = substr(fecha_inicio,5,2)
	destring mes_inicio , replace
	gen dia_inicio = substr(fecha_inicio,7,2)
	destring dia_inicio , replace
	
	gen año_fin = substr(fecha_fin,1,4)
	destring año_fin , replace
	gen mes_fin = substr(fecha_fin,5,2)
	destring mes_fin , replace
	gen dia_fin = substr(fecha_fin,7,2)
	destring dia_fin , replace
	
	gen fechainicio=mdy(mes_inicio,dia_inicio,año_inicio)
	gen fechafin=mdy(mes_fin,dia_fin,año_fin)
	format fechainicio fechafin %td
	drop if año_inicio<2018
	
	gen duracion= fechafin - fechainicio
	gen ruc_contratista = string(ruc, "%15.0f")
	
	collapse (count) n_inhabilitaciones_acum=año_inicio (mean) duracion_inhabilitacion_prom=duracion, by(ruc_contratista)
	label var n_inhabilitaciones_acum "Número acumulado de inhabilitaciones resueltas entre 2018 y 2022"
	label var duracion_inhabilitacion_prom "Duración promedio de las inhabilitaciones resueltas entre 2018 y 2022"
	
	
	save "$bd1/contratistas_inhabilitaciones" , replace
	
	* Sanciones multa
	import delimited "$bd0\sancionados_multa.csv", varnames(1) clear
	
	tostring fecha_inicio, replace
	tostring fecha_fin, replace

	gen año_inicio = substr(fecha_inicio,1,4)
	destring año_inicio , replace
	gen mes_inicio = substr(fecha_inicio,5,2)
	destring mes_inicio , replace
	gen dia_inicio = substr(fecha_inicio,7,2)
	destring dia_inicio , replace
	
	gen año_fin = substr(fecha_fin,1,4)
	destring año_fin , replace
	gen mes_fin = substr(fecha_fin,5,2)
	destring mes_fin , replace
	gen dia_fin = substr(fecha_fin,7,2)
	destring dia_fin , replace
	
	gen fechainicio=mdy(mes_inicio,dia_inicio,año_inicio)
	gen fechafin=mdy(mes_fin,dia_fin,año_fin)
	format fechainicio fechafin %td
	drop if año_inicio<2018
	
	destring monto, replace
	
	gen duracion= fechafin - fechainicio
	
	gen ruc_contratista = string(ruc, "%15.0f")
	
	preserve
	collapse (count) n_multas_acum=año_inicio (mean) duracion_multas_prom=duracion (sum) monto_acum=monto, by(ruc_contratista)
	tempfile datos_acumulados
	save `datos_acumulados'
	restore
	
	collapse (count) n_multas=monto (sum) monto, by(ruc_contratista año_inicio)

	collapse (sum) monto (count) n_multas=monto , by(año_inicio ruc_contratista) 
	collapse (mean) monto_multas_prom=monto n_multas_prom=n_multas, by(ruc_contratista)
	merge 1:1 ruc_contratista using "`datos_acumulados'"
	drop _m
	
	label var monto_multas_prom  "Monto promedio de las multas recibidas entre 2018 y 2022"
	label var n_multas_prom "Número promedio anual de multas recibidas entre 2018 y 2022"
	label var n_multas_acum "Número acumulado de multas recibidas entre 2018 y 2022"
	label var duracion_multas_prom "Duración promedio de las multas recibidas entre 2018 y 2022"
	label var monto_acum "Monto acumulado de las multas recibidas entre 2018 y 2022"
	
	
	save "$bd1/contratistas_multas" , replace

	*Unión de base de datos de contratistas
	
	use "$bd1/bdfinal.dta", clear
	
	
	bysort ruc_contratista ruc_entidad : gen aux=_n
	replace aux=0 if aux!=1
	
	collapse (sum) n_entidades=aux monto_uit_prov=monto_uit (count) n_ordenes=monto_uit , by(anio ruc_contratista)
	collapse (mean) n_entidades_prom=n_entidades monto_prom_anual=monto_uit_prov n_ordenes_prom_anual=n_ordenes (sum) monto_uit_prov, by(ruc_contratista)
	
	destring ruc_contratista, replace
	
	merge 1:1 ruc_contratista using "$bd1/contratistas_penalidades" 
	drop if _m==2
	drop _m
	
	merge 1:1 ruc_contratista using "$bd1/contratistas_inhabilitaciones" 
	drop if _m==2
	drop _m	
	
	merge 1:1 ruc_contratista using "$bd1/contratistas_multas" 
	drop if _m==2
	drop _m	
	
	*Dummy si alguna vez han recibido penalidad, inhabilitacion o multa
	gen penalidad=(n_penalidades_acum!=.)
	label var penalidad "Dummy si ha recibido penalidad entre 2018 y 2022"
	gen inhabilitacion=(n_inhabilitaciones_acum!=.)
	label var inhabilitacion "Dummy si ha recibido inhabilitacion entre 2018 y 2022"
	gen multas=(n_multas_acum!=.)
	label var multas "Dummy si ha recibido multas entre 2018 y 2022"
	gen contratista_dudoso1=(penalidad==1 | inhabilitacion==1 | multas==1)
	label var contratista_dudoso1 "Dummy si el contratista ha recibido penalidad, multa o inhabilitación entre 2018 y 2022"
	
	*Dummy porcentaje de multas respecto monto contratado mayor a 10%
	replace monto_penalidades_acum=0 if monto_penalidades_acum==.
	replace monto_acum=0 if monto_acum==.
	gen pct_multas_penalidades = (monto_penalidades_acum + monto_acum)/monto_uit_prov
	gen contratista_dudoso2=(pct_multas_penalidades>0.1)
	label var contratista_dudoso2 "Dummy si los montos de penalidad y multas recibidas por el contratista superan el 10% del total del monto de sus contratos"	
	
	keep ruc_contratista monto_penalidades_prom n_penalidades_prom monto_penalidades_acum n_penalidades_acum n_inhabilitaciones_acum duracion_inhabilitacion_prom monto_multas_prom n_multas_prom n_multas_acum duracion_multas_prom monto_acum penalidad inhabilitacion multas contratista_dudoso1 contratista_dudoso2
	save "$bd1/basecontratistas.dta", replace
	
	}	
	
*	6. Data final

	use "$bd1/bdfinal.dta" , clear
	
	merge m:1 ruc_contratista using "$bd1/basecontratistas.dta"
	drop if _merge!=3
********************************************************************************
	
*	6. Gráficos iniciales
{	
	graph twoway (histogram monto_uit if monto_uit < 25 & monto_uit >= 2 & objetocontractual == "BIENES" & dummy_2022 ==1, fcolor(gray) lcolor(black) xlabel(2(2)25) xmtick(, labels valuelabel noticks nogrid) name(graph_bienes_2022, replace) xtitle("Monto UIT") ytitle("Frequency") title("Bienes - 2022") ) (scatteri 0 9 1 9, c(l) m(i), lcolor(red), legend(off)) (scatteri 0 8 1 8, c(l) m(i), lcolor(red), legend(off)) 

	graph twoway (histogram monto_uit if  monto_uit < 25 & monto_uit >= 2 & objetocontractual == "BIENES" & dummy_2022 ==0, fcolor(gray) lcolor(black) xlabel(2(2)25) xmtick(, labels valuelabel noticks nogrid) name(graph_bienes_otros, replace) xtitle("Monto UIT") ytitle("Frequency") title("Bienes - Otros")) (scatteri 0 9 1 9, c(l) m(i), lcolor(red), legend(off)) (scatteri 0 8 1 8, c(l) m(i), lcolor(red), legend(off))

	graph twoway (histogram monto_uit if  monto_uit < 25 & monto_uit >= 2 & objetocontractual == "SERVICIOS" & dummy_2022 ==1, fcolor(gray) lcolor(black) xlabel(2(2)25) xmtick(, labels valuelabel noticks nogrid) name(graph_servicios_2022, replace) xtitle("Monto UIT") ytitle("Frequency") title("Servicios - 2022")) (scatteri 0 9 1 9, c(l) m(i), lcolor(red), legend(off)) (scatteri 0 8 1 8, c(l) m(i), lcolor(red), legend(off))

	graph twoway (histogram monto_uit if  monto_uit < 25 & monto_uit >= 2 & objetocontractual == "SERVICIOS" & dummy_2022 ==0, fcolor(gray) lcolor(black) xlabel(2(2)25) xmtick(, labels valuelabel noticks nogrid) name(graph_servicios_otros, replace) xtitle("Monto UIT") ytitle("Frequency") title("Servicios - Otros")) (scatteri 0 9 1 9, c(l) m(i), lcolor(red), legend(off)) (scatteri 0 8 1 8, c(l) m(i), lcolor(red), legend(off))	

	graph twoway (histogram monto_uit if  monto_uit < 25 & monto_uit >= 2 & dummy_2022 ==0, fcolor(gray) lcolor(black) xlabel(2(2)25) xmtick(, labels valuelabel noticks nogrid) name(graph_bs_ss_otros, replace) xtitle("Monto UIT") ytitle("Frequency") title("Todos - Otros")) (scatteri 0 9 1 9, c(l) m(i), lcolor(red), legend(off)) (scatteri 0 8 1 8, c(l) m(i), lcolor(red), legend(off))

	graph twoway (histogram monto_uit if  monto_uit < 25 & monto_uit >= 2 & dummy_2022 ==1, fcolor(gray) lcolor(black) xlabel(2(2)25) xmtick(, labels valuelabel noticks nogrid) name(graph_bs_ss_2022, replace) xtitle("Monto UIT") ytitle("Frequency") title("Todos - 2022")) (scatteri 0 9 1 9, c(l) m(i), lcolor(red), legend(off)) (scatteri 0 8 1 8, c(l) m(i), lcolor(red), legend(off))

	graph combine graph_bienes_otros graph_bienes_2022  graph_servicios_otros graph_servicios_2022 graph_bs_ss_otros graph_bs_ss_2022, cols(2) ///
    title("Distribución de compras según año y tipo de objeto", size(large))
	
}	