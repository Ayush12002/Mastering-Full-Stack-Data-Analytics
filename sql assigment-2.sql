use mavenmovies;
-- Q 1 -
select * from film
where length between 50 and 100;
select * from actor
limit 50;
select distinct flim_id from inventory;
select count(*) as total_rentals
from rental;
-- Q2-
select avg (rental_duration) as average_rental_duration 
from film ;
-- Q 3-
select upper(first_name) as first_name, upper(last_name) as last_name 
from customer;

-- Q 4-
select rental_id, month (rental_date) as rental_months
from rental;

-- Q 5-
select customer_id, count(*) as rental_count
from rental 
group by customer_id;

-- Q 6-
select * from store ;
select amount, sum(amount) as total_revenue
from payment
 group by amount;
 -- Q 7-
 select a.first_name, a.last_name
 from actor a
 join film_actor fa on a.actor_id = fa.actor_id 
 join film f on fa.film_id = f.film_id 
 where f.title = 'gone with the wind';

-- Q 8-
SELECT c.name as category, COUNT(r.rental_id) AS total_rentals
FROM category c
JOIN film_category fc ON c.category_id = fc.category_id
JOIN inventory f ON fc.film_id = f.film_id
left JOIN rental r ON f.inventory_id = r.inventory_id
GROUP BY c.name;
-- Q 9-
SELECT l.name AS language, AVG(f.rental_rate) AS average_rental_rate
FROM language l
JOIN film f ON l.language_id = f.language_id
GROUP BY l.name;
-- Q 10-
select c.first_name, c.last_name , sum(p.amount) as total_spend
from customer c
join payment p on c.customer_id = p.customer_id
group by c.first_name , c.last_name;

-- Q 11-
select c.first_name, c.last_name , f.title
from customer c
join address a on c.address_id = a.address_id
join city ci on a.city_id = ci.city_id
join rental r on c. customer_id = r.customer_id 
join inventory i on r.inventory_id = i.inventory_id
join film f on i.film_id = f.film_id
where ci.city = 'london';
-- Q 12 -
select f.title, count(*) as rental_count
from film f
join inventory i on f.film_id = i.film_id
join rental  r on i.inventory_id = r.inventory_id
group by f.title 
order by rental_count desc
limit 5;
-- Q 13-
SELECT c.customer_id, c.first_name, c.last_name
FROM customer c
JOIN rental r ON c.customer_id = r.customer_id
JOIN inventory i ON r.inventory_id = i.inventory_id
JOIN store s ON i.store_id = s.store_id
WHERE s.store_id IN (1, 2)
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING COUNT(DISTINCT s.store_id) = 2;