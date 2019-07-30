# Adaptive-Sequential-Sampling
Requirements: 
1. Commercial optimization software CPLEX  
2. Open-source numerical computing package Eigen

## How to Run CPLEX on Windows

##### Additional Requirements
Install Visual Studio 2019
- After installing, choose **Desktop development with C++** workload

##### Create a Project
1. Once Visual Studio finished installing, click on **Create a new project**
2. Click on **Empty Project** with tabs C++, Windows, Console and create a Project Name
3. Now, add all the files above
   - Project >> Add Existing Item >> Select your files
    - Appropriate files will be sorted into the Header Files and Source Files
4. Next, you have to set some options so that the project knows where to find the CPLEX and Concert include files and CPLEX, Concert, and Eigen libraries.

From the **Project** menu, choose **<Project Name> Properties**.The **<Project Name> Property Pages** dialog box appears.
  
   1. In the **Configuration** drop-down list, select **Release**.In the **Platform** drop-down list, select **x64** to create a 64-bit application.

   2. Select **C/C++** in the **Configuration Properties** tree.
      - Select **General**:
        - In the **Additional Include Directories** field, add the directories:
          - C:\Program Files\IBM\ILOG\CPLEX_Studio1290\cplex\include
          - C:\Program Files\IBM\ILOG\CPLEX_Studio1290\concert\include
          - Path to the eigen folder
            - E.g. C:\Users\<Username>\downloads\eigen-eigen-323c052e1731
      - Select **Preprocessor**:
        - Add IL_STD to the **Preprocessor Definitions** field.
      - Select **Code Generation**:
        - Set **Runtime Library** to **Multi-threaded DLL (/MD)**.
   3. Select **Linker** in the **Configuration Properties** tree.
      - Select **General** and then select **Additional Library Directories. Add the directories :
        - C:\Program Files\IBM\ILOG\CPLEX_Studio1290\cplex\include\lib\x64_windows_vs2015\stat_mda
        - C:\Program Files\IBM\ILOG\CPLEX_Studio1290\concert\lib\x64_windows_vs2015\stat_mda
      - Select **Input** and then select **Additional Dependencies**. Add the files:
        - cplex1290.lib
        - ilocplex.lib
        - concert.lib

Click **OK** to close the **<Project Name> Property Pages** dialog box.

3. Next, you have to set the default project configuration. From the **Build** menu, select **Configuration Manager…**
   - Select **Release** in the **Active Solution Configuration** drop-down list.
     - Select **x64** in the **Active Solution Platforms** drop-down list.
     - Click **Close**.

##### To Build/Compile the CPLEX Project
Finally, to build the project, from the **Build** menu, select **Build Solution**.
After completion of the compiling and linking process, the target is created, ** <project name>.exe**.
 
##### To Run the CPLEX Project
Once a successful build is made, you are ready to run your program:

1. Tools >> Visual Studio Command Prompt
2. The program consists of 7 command line arguments:
   1. Path to executable file  
      - The full path of the **<project name>.exe** is typically the second to last line of the **Output Box** after a successful build.
   2. Path to instance  (folder provided above)
   3. Path to a results folder
      - The output of the program is a text file that print the results of the chosen option by the user.
   4. Option 
   5. ---
   6. ---
   7. Random number

E.g. of command line arguments
"C:\Users\<username>\source\repos\<project name>\x64\Release\<project name.exe" "C:\Users\<username>\source\repos\<project name>\instances\20x20-1-20000-1-clean.dat" "C:\Users\<username>\source\repos\<project name>\results\temp" 0 0 0.5 1e-3 5
