-- Specific Case Questions and Queries
-- Problem Statement: Find departments where number of employees earning over $50,000 is greater than 10.
-- Calculate total salary for these departments.
SELECT department, COUNT(employee_id) AS num_employees, SUM(salary) AS total_salary
FROM employees
WHERE salary > 50000
GROUP BY department
HAVING COUNT(employee_id) > 10;

-- ISODOW Function
-- Problem Statement: Determine the day of the week each order was placed using ISO standard.
SELECT order_id, order_date, EXTRACT(ISODOW FROM order_date) AS weekday
FROM orders;

-- NTILE Function
-- Problem Statement: Divide employees into 4 salary-based quartiles.
SELECT employee_id, salary, NTILE(4) OVER (ORDER BY salary DESC) AS salary_quartile
FROM employees;

-- WINDOW Functions
-- Problem Statement: Calculate the average salary within each department.
SELECT employee_id, department, salary, AVG(salary) OVER (PARTITION BY department) AS avg_department_salary
FROM employees;

-- EXTRACT Date/Time Values
-- Problem Statement: Extract the year and month from the order date for analysis.
SELECT order_id, EXTRACT(YEAR FROM order_date) AS order_year, EXTRACT(MONTH FROM order_date) AS order_month
FROM orders;

-- Time-Series Data Analysis
-- Problem Statement: Analyze the month-over-month growth rate in sales. 
-- Calculate total sales for each month and compute the percentage change from the previous month.
WITH monthly_sales AS (
  SELECT EXTRACT(YEAR_MONTH FROM sale_date) AS year_month, SUM(amount) AS total_sales
  FROM sales
  GROUP BY year_month
)
SELECT year_month, 
       total_sales,
       (total_sales - LAG(total_sales) OVER (ORDER BY year_month)) / LAG(total_sales) OVER (ORDER BY year_month) * 100 AS growth_rate
FROM monthly_sales;

-- Complex JOIN Operations
-- Problem Statement: Identify employees who have not completed any of the mandatory training courses.
SELECT e.employee_id, e.name
FROM employees e
LEFT JOIN completed_trainings ct ON e.employee_id = ct.employee_id
WHERE ct.course_id IS NULL;

-- Recursive Queries
-- Problem Statement: Create an organizational hierarchy tree.
WITH RECURSIVE hierarchy AS (
  SELECT employee_id, name, manager_id
  FROM employees
  WHERE manager_id IS NULL
  UNION ALL
  SELECT e.employee_id, e.name, e.manager_id
  FROM employees e
  INNER JOIN hierarchy h ON e.manager_id = h.employee_id
)
SELECT * FROM hierarchy;

-- Advanced Aggregation with Filtering
-- Problem Statement: Calculate the average sales amount for top-performing employees in each department.
SELECT department, AVG(s.total_sales) AS avg_sales
FROM (
  SELECT employee_id, SUM(amount) AS total_sales,
         NTILE(10) OVER (PARTITION BY department ORDER BY SUM(amount) DESC) AS performance_percentile
  FROM sales
  GROUP BY employee_id
) s
WHERE s.performance_percentile = 1
GROUP BY department;

-- Complex Data Transformation
-- Problem Statement: Identify user sessions with continuous activity for more than 1 hour.
WITH sessions AS (
  SELECT user_id, 
         activity_start, 
         LEAD(activity_start) OVER (PARTITION BY user_id ORDER BY activity_start) AS next_activity_start
  FROM user_activities
)
SELECT user_id, activity_start, next_activity_start
FROM sessions
WHERE TIMESTAMPDIFF(MINUTE, activity_start, next_activity_start) <= 5;


