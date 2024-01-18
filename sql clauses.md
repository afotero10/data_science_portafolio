## SQL Clauses

### SELECT and FROM

The foundation of any SQL query lies in the SELECT and FROM clauses. Ensure you are comfortable with selecting specific columns and joining tables to retrieve the necessary data.

### WHERE and HAVING

The WHERE clause is used to filter rows based on specified conditions, while the HAVING clause is used with aggregate functions to filter groups. Understand the difference between them and when to use each in your queries.

## SQL Functions

### ISODOW Function

The ISODOW function returns the ISO day of the week for a given date, where Monday is considered the first day of the week (1) and Sunday is the last day (7). Familiarize yourself with its usage in date-related queries.

### NTILE Function

The NTILE function is used to divide the result set into a specified number of roughly equal parts. It is useful for ranking and distribution analysis. Be prepared to use NTILE in scenarios where data needs to be distributed into buckets.

### WINDOW Functions

WINDOW functions operate on a set of table rows related to the current row. Examples include ROW_NUMBER(), RANK(), and DENSE_RANK(). Understand how to apply these functions to create advanced analytical queries.

### EXTRACT Date/Time Values

The EXTRACT function is used to retrieve specific parts of a date or time, such as HOUR, YEAR, MONTH, etc. Be ready to incorporate EXTRACT into your queries to extract relevant information from datetime columns.

### AVG/SUM Functions

Aggregate functions like AVG (average) and SUM (sum) are crucial for performing calculations on groups of rows. Know how to use these functions in conjunction with GROUP BY to derive meaningful insights from your data.

## WHERE/HAVING Clause

Differentiating between the WHERE and HAVING clauses is essential. WHERE filters individual rows before grouping, while HAVING filters grouped rows based on conditions involving aggregate functions. Practice scenarios where both clauses are necessary.
