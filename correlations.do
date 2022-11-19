clear 
 import delimited using /Users/ddemszky/classroom-transcript-analysis/data/transcript_metadata_with_predictions.csv, bindquotes(strict) maxquotedrows(100)
 
	set more off
	//cd "/Volumes/GoogleDrive/My Drive/Teacher Uptake/"
	glob date = c(current_date)

  // dependent variables
 global depvars mqi5 clinstd clts clrsp clpc
 
 // Standardize variables
 foreach var in $depvars  {
	 	egen z_`var' = std(`var')
 }
		
		
 foreach var in student_reasoning student_on_task  {
	 	gen `var'_prop = `var'_pred / num_chapters
	 }
	 
	  foreach var in teacher_on_task focusing_question high_uptake  {
	 	gen `var'_prop = `var'_pred / num_chapters
	 }
 
 // predictions
 global predvars student_reasoning_prop teacher_on_task_prop student_on_task_prop focusing_question_prop high_uptake_prop student_turn_prop student_word_prop
 

 // teacher covariates
 global teacher_covars male black white asian hisp experience raceother
 
 // student covariates
 global student_covars s_male s_afam s_white s_hisp s_asian s_race_other s_frpl s_sped s_lep
  
  

	 
  
  eststo clear
foreach predvar in $predvars {
	foreach var in $depvars {
		eststo: regress z_`var' `predvar' $teacher_covars $student_covars, vce(cluster nctetid)
	}
   
}

collapse (mean) stateva_obs_year $depvars $predvars $teacher_covars $student_covars, by(nctetid year)

egen z_stateva_obs_year = std(stateva_obs_year)

foreach predvar in $predvars {
	regress z_stateva_obs_year `predvar' $teacher_covars $student_covars, vce(cluster nctetid)
	
   
}



   
   esttab using "/Users/ddemszky/classroom-transcript-analysis/results/corrs_$predvar_$date.csv", ///
   label se( 3 ) b( 3 ) star( + 0.10 * 0.05 ** 0.01 ) nonote  ///
   keep(student_reasoning_prop teacher_on_task_prop student_on_task_prop focusing_question_prop high_uptake_prop student_turn_prop student_word_prop) ///
   stats(r2 N , fmt( 3 0) label( "R^2" "Observations")) varwidth(25) ///
   nogaps replace
