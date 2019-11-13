SRS Meetings application
-----------------------

Dash application to display different KPIs per subteam for the weekly SRS Meetings. The main aim of this project is the application itself but in order to have it running, please make sure you have a consistent database for the KPI (detailed below).

Installation
----------------------

#### Before starting

This application uses *New Relic* data in order to compute the dashboard. Hence first make sure you have a *New Relic* account. After you will need to generate an API key in the general settings in order to make requests directly from the scripts. Also ensure you have your account ID.

In order to compute the dashboard, you should have settled a MySQL database and have in mind the followings : host name, user name, password, database name, table name. Details on where to put those information in the code will be explained further in this document.
#### Settings

* Clone this repo to your computer by using `git clone https://blini_circles@bitbucket.org/libertywireless/circles-srs-stats.git` in the folder you want the repo to be in.
* Get into the folder using `cd circles-srs-stats`.
* Inside the folder `private`, create a `private.py` file. This file should contain several parameters that you need to set up before using the scripts.
    * Information to access the New Relic database : API_KEY_NR and ACCOUNT_ID
    * Information to access the JIRA dashboard : USER_JIRA, API_KEY_JIRA, SERVER_JIRA, JIRA_PROJECT
    * Parameters to access the MySQL database : HOST_MYSQL, USER_MYSQL, PASSWORD_MYSQL, DATABASE_MYSQL, TABLE_MYSQL
    * Parameters to launch on specific server : HOST, PORT
    * OpsGenie API : API_KEY_OPS_GENIE
    * Saving paths : JIRA_CSV_SAVE_PATH, KPI_PER_WEEK_SAVE_PATH, API_PER_WEEK_SAVE_PATH, TEMPLATE_PER_WEEK_SAVE_PATH,
                TICKETS_PER_WEEK_SAVE_PATH, API_USER_FACING_SAVE_PATH, API_TO_CUSTOMER_FACING_SAVE_PATH,
                API_LEVEL_MEDIAN_SAVE_PATH
    * EXPECTED_API_HIGHER_LATENCIES
    * DIR_PATH
    * API_TO_CUSTOMER_FACING_REL_PATH
    * TEAM_MEMBERS
* Add this `private.py` file to the .gitignore so Git will ignore this when you push your code.


#### Install the requirements

* Make sure you have a correct environment created for this project. You might want to use a virtual environment.  Python 3 preferred. In any case, activate your environment.
 
* Install the requirements using `pip install -r requirements.txt`.
    * If using Python 2, remove the two # in the requirements.txt file

* Install the setup file using `python setup.py install`

Usage
-----------------------

* By default, if you run your script on a given day, the KPIs will be computed for a week ending the previous day at 11:59:59 PM. If you want to change those settings (for testing for example), you have to update `BEGIN_DATE_WEEKLY`, `END_DATE_WEEKLY`, `BEGIN_DATE_FORMER_WEEK`, `END_DATE_FORMER_WEEK` in [connect.py]((./ressources/connect.py)).
* To run main scripts, go to the src.run folder.
* Run `update_db.py` to locally create a database for the KPIs. Each line corresponds to all KPIs for a given subteam and week.
* Run `srs_meetings_application.py` to locally launch the application on your computer.
* Run `main_run_weekly.py` to launch and save all the necessary information for the SRS meetings. You can change the booleans in order to run only some specific parts of the code. (most important function, for now the application is unused.)

Describing project architecture
----------------------------------

* [private](./private)
    * [private](./private/private.py) : private file where you set up your parameters.
* [ressources](./ressources)
     * [get_former_data](./ressources/get_former_data) : mapping Excel strings to denominations used in the Python code. Eventually easier to create the database manually hence unused.
     * [get_query_kpi](./ressources/get_query_kpi)
       * [generating_query](./ressources/get_query_kpi/generating_query.py) : generating NRQL queries depending on wanted KPI to compute in the *New Relic* database.
       * [response_type_to_kpi](./ressources/get_query_kpi/response_type_to_kpi.py) : depending on response type from the API request, path to find the interesting number.
     * [connect](./ressources/connect.py) : parameters manually modified in the `private.py` file and imported. If cannot load parameters simply pass.
     * [team_to_kpi](./ressources/team_to_kpi.py) : information on KPIs and subteam.
* [src](./src)
    * [create_db](./src/create_db)
        * [get_kpi_per_subteam](./src/create_db/get_kpi_per_subteam.py) : for each subteam enrich a dictionary to store the KPI information for a given week. 
        * [pre_check_for_db](./src/create_db/pre_check_for_db.py) : before adding new data to database, checking list to be done. For now just checking it has not been less than a week.
    * [dash_application](./src/dash_application)
        * [layouts](./src/dash_application/layouts)
            * [comparison_layout](src/dash_application/layouts_/comparison_layout.py) : layout to display graphs to compare all sub teams.
            * [daily_table_layout](src/dash_application/layouts_/daily_table_layout.py) : layout to display KPIs from day to day, not used for now.
            * [dashboard_layout](src/dash_application/layouts_/dashboard_layout.py) : layout to display the home page to choose both a sub team and a week for the KPIs.
            * [full_table_srs_meetings_layout](src/dash_application/layouts_/full_table_srs_meetings_layout.py) : layout to display the current format of table used in the Gsheet for SRS meeting.
            * [sub_team_graph_layout](src/dash_application/layouts_/sub_team_graph_layout.py) : layout to display graphs for one given sub team.
        * [full_table_fct](./src/dash_application/full_table_fct.py) : functions to help display the whole table with all info for subteams.
        * [generator_functions](./src/dash_application/generator_functions.py) : generating dash core components elements such as dropdowns and 
        * [helper_functions](./src/dash_application/helper_functions.py)  : helper functions for the dash application
        * [parameters](./src/dash_application/parameters.py) : specific parameters for the dash application.
    * [run](./src/run)
        * [create_jira_tickets](./src/run/create_jira_tickets.py) : Creating JIRA tickets automatically. Not used anymore.
        * [generate_api_latency_per_week](./src/run/generate_api_latency_per_week.py) : Calculating average latency for each API for each sub team, with a threshold to determine which APIs to keep.
        * [generate_kpi_info_per_week](./src/run/generate_kpi_info_per_week.py) : run this to get the dataframe of the KPIs for the past week.
        * [generate_text_confluence](./src/run/generate_text_confluence.py) : Automatically create the text for Confluence action items. In progress.
        * [generate_tickets_per_week](./src/run/generate_tickets_per_week.py) : Calculating number of tickets (OpsGenie /  CS Escalations) created for each team per week.
        * [main_run_weekly](./src/run/main_run_weekly.py) : Main script to run to get data for the SRS meeting.
        * [srs_meeting_application](./src/run/srs_meetings_application.py) : run this to launch locally the Dash application.
        * [update_db_srs_meeting](src/run/update_db_srs_meeting.py) : run this to automatically update the database.
    * [tickets_from_channel](./src/tickets_from_channels)
        * [count_per_team](./src/tickets_from_channels/count_per_team.py) : Couting numbers of tickets/alerts created by team.
        * [helper_functions_tickets](./src/tickets_from_channels/helper_functions_tickets.py) : helper functions.
        * [jira_api_tickets](./src/tickets_from_channels/jira_api_tickets.py) : for JIRA tickets.
        * [opsgenie_api_tickets](./src/tickets_from_channels/opsgenie_api_tickets.py) : for OpsGenie alerts.

* [.gitignore](./.gitignore) : files to be ignores by Git when committing 
* [API_to_customer_facing](./API_to_customer_facing.csv) : mapping APIs to customer facing or not.
* [kpi_per_week](./kpi_per_week.db) : database with all the KPIs
* [kpi_per_week_fake_for_test](./kpi_per_week_fake_for_test.db) : manually created database for testing on the application.
* [README.md](./README.md)
* [README_bitbucket](./README_bitbucket.md) : general information for Bitbucket.
* [requirements.txt](./requirements.txt#)


How to add a new KPI for a given subteam (if the KPI requires only one query)
------------------------------------------

If you want to add a new KPI for a given subteam (whether the subteam previously existed or not in the database), you have to ensure that running `generate_kpi_info_per_week.py` with the right parameters indeed creates a dataframe with info on the newly added KPI.

The mentioned script uses one function, `generate_df`, for which you will need to complete six parameters.
* ``api_key`` : already mentioned before
* ``account_id`` : already mentioned before
 
* Check if the KPI exists or not in the ALL_KPIS from [team_to_kpi.py](./ressources/team_to_kpi.py). If not, you should add it by following a structure similar to the keys already existing. Put the KPI name for the **key** and a dict of attributes for the **value.**
    * _response_type_ : to be checked directly on the New Relic Online Query. Most of the time the json response on New Relic will help you set this parameter.
    * _to_get_fct_ : function which computes the NRQL query. If your function is _f_ structure should be _gf.f_ and you should add the corresponding function in [generating_query.py](./ressources/get_query_kpi/generating_query.py)
    * _measure_unit_ : measure unit of the KPI. It serves only to display it on the graphs.

* Check if the response type entered before exists or not in the RESPONSE_TYPE_TO_KPI from [response_type_to_kpi.py](./ressources/get_query_kpi/response_type_to_kpi.py). If not, add it on this dictionary by finding the path to get the wanted number in the output of the NRQL query.
* Check if the team and the sub team for which we want to calculate the KPI exists or not in the TEAM_TO_SUBTEAM from [team_to_kpi.py](./ressources/team_to_kpi.py). If not, add the corresponding key and value in the dictionary.
* Check if the sub team info exists or not in the SUBTEAM_INFO from [team_to_kpi.py](./ressources/team_to_kpi.py). 
    * If it already existed, do not forget to add the new KPI in the already existing list of KPIs.
    * If it does not exist, put the sub team name as **key** and a dict of attributes as **value**.
        * _KPI_: list of KPIs to be calculated for the sub team
        * _params_: list of params (in the right order) to generate the NRQL query
        * each element in the params should be a key of the dictionary, with the real value as value.
    
How to add a new KPI for a given subteam (if the KPI requires two queries and a division)
------------------------------------------

For some KPIs it may require fetching numbers from two different databases, hence as it is not supported by New Relic yet you will have to compute it with two queries. For now the operation available is division but is of course subject to change. 

* First ensure you have the two NRQL queries you want to compute in the right order (first the numerator and then the denominator)
* Add those two queries by adding two functions in the [generating_query.py]((./ressources/get_query_kpi/generating_query.py)) if a similar query is not already entered, and put it in the right format. (See examples in the given python file)
* Change the `ALL_KPIS` in the [team_to_kpi.py]((./ressources/team_to_kpi.py)) file.
    * If the KPI is a specific KPI of an already existing KPI, add a key, value similar to MNOMW in the availability specific dictionary.
    * It this is a new KPI, add a key, value similar to crash_free_sessions.
* Change the `SUBTEAM_INFO` in the [team_to_kpi.py]((./ressources/team_to_kpi.py)) file. (See MNOMW for example in the dictionary)
* Change the `enrich_dict_kpi` in the [get_kpi_per_subteam.py]((./src/create_db/get_kpi_per_subteam.py)) python file. 
