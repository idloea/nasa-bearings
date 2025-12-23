# README #

Repository for analysing bearing failures from the NASA bearings dataset. 

### What is this repository for? ###

* This repository analyses bearing failures using vibration data.

### How do I get set up? - WIP ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions
* Dataset set up
  1. Download the NASA IMS bearings in ZIP format: https://data.nasa.gov/dataset/ims-bearings
  2. Create a folder named `data`in the repository:
  3. Extract downloaded ZIP file inside `data`
  4. It is recommended to rename the folder names to follow the next structure:
  
  
      ```
      data/
      └── nasa_ims_bearing_dataset/
          ├── 1st_test/
          │   └── 1st_test/
          ├── 2nd_test/
          │   └── 2nd_test/
          └── 3rd_test/
              └── 4th_test/
      ```     
       Note: In the NASA source, the subfolder within `3rd_test` is labeled `4th_test`. Keep this naming convention to ensure the data loading scripts function correctly.
     
### Contribution guidelines - WIP  ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? - WIP ###

* Repo owner or admin
* Other community or team contact
