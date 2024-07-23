--DATA CLEANING IN SQL--

---Step 1: Provider_CCN values must display 6 digits by adding zero to the left side---
---Step 2: Fiscal Dates must be converted from Excel character value to SQL date value---
---Step 3: Remove duplicate values under hospital_name column and only keep most recent count---
	---Step 3a: Create Common Table Expression of updated table---
	---Step 3b: Add partition statement to query to add numbered values to most recent count of beds---
---Step 4: Apply Steps 1 & 2 to the hcahps_data table---
---Step 5: Join tables using provider_ccn value---


with hospital_beds_p as
(
select lpad(cast(provider_ccn as text),6,'0') as provider_ccn,
	   hospital_name,
	   to_date(fiscal_year_begin_date,'MM/DD/YYYY') as fiscal_year_begin_date,
	   to_date(fiscal_year_end_date,'MM/DD/YYYY') as fiscal_year_end_date,
	   number_of_beds,
	   row_number() over (partition by provider_ccn order by to_date(fiscal_year_end_date,'MM/DD/YYYY') desc) as recent_row
from "postgres"."Hospital_Data".hospital_beds
)

select lpad(cast(facility_id as text),6,'0') as provider_ccn,
	   to_date(start_date,'MM/DD/YYYY') as begin_date_converted,
	   to_date(end_date,'MM/DD/YYYY') as end_date_converted,
	   hcahps.*,
	   hosp_beds.number_of_beds,
	   hosp_beds.fiscal_year_begin_date as reported_start_date,
	   hosp_beds.fiscal_year_end_date as reported_end_date
from "postgres"."Hospital_Data".hcahps_data as hcahps
left join hospital_beds_p as hosp_beds
	on lpad(cast(facility_id as text),6,'0') = hosp_beds.provider_ccn
and hosp_beds.recent_row = 1