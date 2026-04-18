understand the way i have set up this directory. im trying to complete a personal project detailed in @2D-IDEAL-MHD-FVM-SOLVER.md. i want   
to build a python solver, build latex documentation to solve the       
orztag vortex problem as detailed in                                                        
https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html. You       
should follow industry-standard python standards, use git source control to store           
incremental progress (don't push but you can commit). I have Initialized git, but I haven't added any commits yet. This is because I want you to have access to the references directory. you should use the .gitignore in the .scratch directory before your first commit.
                                                                                            
- Build some architecture to input a mesh (same problem as the vortex MHD)                  
- take that mesh and construct a grid                                                       
- build in the MHD equations                                                                
- solve and use proper logging and outputing iteration to some log file                     
- get incremental results and properly check those results                                  
- you can jump straight to the 2D problem, and skip the 1D problem                          
- after the solver works, document the creation in latex with important sections that match the rubric in @references/notes/ for the write up.
- you should adapt/follow the pertinent content in the "cfd-hw" skill for code quality, latex write-up, etc.
- KEEP IN MIND that i am on a simple system and dont have a lot of compute so choose your mesh and residuals to be proportionate to my compute. So don't make the mesh too fine and all that.
- when you're done with the code/solver build out and the latex documentation, make another folder called "presentation" that uses a simple latex beamer template and documents everything in "references/notes/presentation-rubric.pdf".
- when creating the latex docs, you should reference the textbooks in the "2D-IDEAL-MHD-FVM-SOLVER.md" when required.