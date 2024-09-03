use mavenmovies;
select* from actor;
select* from customer;
select *from city;
select * from customer where active +1;
select * from film where rental_duration > 5;
select count(*) as total_flims
from film
where replacement_cost > 15 and replacement_cost <20;
select count(distinct first_name) as unique_first_names
from actor;
select * from customer
limit 10;
select *from coustomer 
where first_name like '8%'
limit 3;
select title from film
where rating = 'G'
limit 5;
select * from customer
where first_name like 'A%';
select * from customer 
where first_name like 'A%';
select city from city 
where city like 'A%A'
limit 4;
select * from customer 
where first_name like '%ni%';
select * from customer 
where first_name like '_r%';
select * from customer 
where first_name like 'a___%';
select * from customer 
where first_name like 'A%o';
select * from film
where rating IN ( 'pg','pg-13' );
select * from film
where length between 50 and 100;
select *from actor
limit 50;
select distinct film_id from inventory;





