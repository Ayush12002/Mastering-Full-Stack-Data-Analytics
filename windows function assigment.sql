use mavenmovies;
-- Q1 -

    
    -- Q2 --
 select f.film_id,
    f.title,
    
    SUM(p.amount)  AS cumulative_revenue
FROM
    film f
JOIN inventory i on f.film_id = i.film_id
   join  rental r ON i.inventory_id = r.inventory_id
   join payment p on r.rental_id = p.rental_id
group by f.film_id, f.title
order by cumulative_revenue desc;
    
    -- Q3 --


    SELECT
        f.film_id,
        f.title,
        AVG(r.rental_duration) AS avg_rental_duration
    FROM
        film f
    JOIN
        rental r ON i.inventory_id = r.inventory_id
        join payment p on r.rental_id = p.rental_id
        
    GROUP BY
        f.film_id, f.title
        order by cumulative_revenue desc;
        

SELECT
    film_id,
    title,
    rental_duration,
    avg(rental_duration) over (partition by rental_duration) as average_duration
FROM
    Film
ORDER BY
    rental_duration;
    
    -- Q4 --
    WITH RentalCounts AS (
    SELECT
        f.film_id,
        f.title,
        fc.category_id,
        COUNT(r.rental_id) AS rental_count
    FROM
        films f
    JOIN
        rentals r ON f.film_id = r.film_id
    JOIN
        film_category fc ON f.film_id = fc.film_id
    GROUP BY
        f.film_id, f.title, fc.category_id
),
RankedFilms AS (
    SELECT
        film_id,
        title,
        category_id,
        rental_count,
        RANK() OVER (PARTITION BY category_id ORDER BY rental_count DESC) AS rank
    FROM
        RentalCounts
)
SELECT
    film_id,
    title,
    category_id,
    rental_count
FROM
    RankedFilms
WHERE
    rank <= 3;
    
    
    -- Q 5 --
WITH CustomerRentalCounts AS (
    SELECT
        customer_id,
        COUNT(*) AS total_rentals
        from rental 
        group by customer_id 
        ),
   

AverageRentals AS (
    SELECT
        AVG(total_rentals) AS avg_rentals
    FROM
        CustomerRentalCounts
)
SELECT
    c.customer_id,
    c.first_name,
    c.last_name, cr.total_rentals,
    cr.total_rentals - ar.avg_rentals AS difference
FROM
    CustomerRentalCounts cr
    join customer c on cr.customer_id = c.customer_id
CROSS JOIN
    AverageRentals ar;
    
    
    -- Q6--
       select year(payment_date) as year, month(payment_date) as months,
       
    SUM(amount) AS total_revenue
FROM payment
    
GROUP BY
year(payment_date), month(payment_date)
ORDER BY year (payment_date), months(payment_date);

-- Q7--

WITH CustomertotalSpending AS (
    SELECT
        customer_id,
        SUM(amount) AS total_spend
    FROM payment
         group by customer_id 
         ),
         rankedcustomers as (
         select customer_id, total_spend,
         ntile(5) over (order by total_spend desc) as spending_rank
         from customertotalspending
         )

SELECT
    c.customer_id,c.first_name,c.last_name, r.total_spend
FROM rankedcustomers r
join customer c on r.customer_id = c.customer_id
    WHERE r.Spending_Rank = 1;

-- Q8 --
with categoryrentalcounts as (
select c.name as category, count(*) as rental_count
from category c
join film_category fc on c.category_id = fc.category_id
join film f on fc.film_id = f.film_id
join inventory i on f.film_id = i.film_id
join rental r on i.inventory_id = r.inventory_id
group by c.name
)
select category, rental_count,
sum(rental_count) over (order by rental_count) as sunning_total
from categoryrentalcounts
order by rental_count;

-- Q9--
WITH filmRentalCounts AS (
    SELECT
        
        f.film_id,
        f.title,
        c.name as category,
        COUNT(*) AS rental_count
    FROM
        film f
    JOIN
        film_category fc on f.film_id = fc.film_id 
    JOIN
        category c on fc.category_id = c.category_id
        join inventory i on f.film_id = i.film_id
        join rental r on i.inventory_id = r.inventory_id
    GROUP BY
     f.film_id, f.title, c.name
),
Categoryavgrentalcounts AS (
    SELECT
        category,
        AVG(rental_count) AS avg_rental_count
    FROM
        filmRentalCounts
    GROUP BY
        category
)
SELECT fr.film_id,
    fr.title,
    fr.category,
    fr.rental_count
   
FROM
    filmRentalCounts fr
JOIN
    Categoryavgrentalcounts ca ON fr.category = ca.category
WHERE
    fr.rental_count < ca.avg_rental_count;

-- Q10 --
select year (payment_date) as year,month(payment_date) as months,
sum(amount) as total_revenue
from payment
group by year(payment_date), month(payment_date)
order by total_revenue desc
limit 5;