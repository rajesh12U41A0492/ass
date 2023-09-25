
use assignment2
create table company_info(emp_id int not null,emp_name varchar(20) not null,emp_position varchar(20) not null,salary int not null);
insert into company_info(emp_id,emp_name,emp_position,salary)
                  values(1234,"john","ceo",1000000),
						(5678,"peter","manager",500000),
                        (9879,"patrik","assistant",30000),
						(5432,"sam","tech",15000);
select * from company_info;
create table empolyee_info(name char (20) not null,fahtername char(20) not null,address varchar(20) not null,qualification char(20) not null, maritalstatus char(20) not null);
insert into empolyee_info(name,fahtername,address,qualification,maritalstatus)
				  values("john","johny","dholakpur","mtech","divorce"),
						("peter","penny","coastarica","btech","married"),
                        ("patrik","bheem","chennai","bcom","single"),
                        ("sam","dholu","kolkata","iti","single");
select * from empolyee_info
select com.emp_id,com.emp_name,com.emp_position,com.salary,empl.name,empl.fahtername,empl.address,empl.qualification,empl.maritalstatus from company_info as com,empolyee_info as empl;
select com.emp_id,com.emp_name,com.emp_position,com.salary,empl.name,empl.fahtername,empl.address,empl.qualification,empl.maritalstatus from company_info as com inner join empolyee_info as empl on com.emp_name=empl.name;
select com.emp_id,com.emp_name,com.emp_position,com.salary,empl.name,empl.fahtername,empl.address,empl.qualification,empl.maritalstatus from company_info as com left join empolyee_info as empl on com.emp_name=empl.name;
select com.emp_id,com.emp_name,com.emp_position,com.salary,empl.name,empl.fahtername,empl.address,empl.qualification,empl.maritalstatus from company_info as com right join empolyee_info as empl on com.emp_name=empl.name;
select * from company_info union all
select * from empolyee_info;
select * from employee_info
where empolyee_info.name in (select customer_info.name from customer_info);
