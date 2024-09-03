use mavenmovies;
-- Q 1-
select* from actor;
-- Q 2-
select* from customer;
-- Q 3-
select *from city;
-- Q 4-
select * from customer where active +1;
-- Q 5-
select * from film where rental_duration > 5;
-- Q 6-
select count(*) as total_flims
from film
where replacement_cost > 15 and replacement_cost <20;
-- Q 7 -
select count(distinct first_name) as unique_first_names
from actor;
-- Q 8-
select * from customer
limit 10;
-- Q 9-
select *from coustomer 
where first_name like '8%'
limit 3;
-- Q 10-
select title from film
where rating = 'G'
limit 5;
-- Q 11-
select * from customer
where first_name like 'A%';
-- Q 12-
select * from customer 
where first_name like 'A%';
-- Q 13-
select city from city 
where city like 'A%A'
limit 4;
-- Q 14-
select * from customer 
where first_name like '%ni%';
-- Q 15-
select * from customer 
where first_name like '_r%';
-- Q 16-
select * from customer 
where first_name like 'a___%';
-- Q 17 -
select * from customer 
where first_name like 'A%o';
-- Q 18-
select * from film
where rating IN ( 'pg','pg-13' );
-- Q 19-
select * from film
where length between 50 and 100;
-- Q 20 -
select *from actor
limit 50;
-- Q 21-
select distinct film_id from inventory;





