program main
  implicit none

  double precision, parameter :: pi = 3.141592653589793238462643383279502884197d0
  integer, parameter          :: w = 10
  integer, parameter          :: iters = 10

  double precision :: data(w), result(w)
  integer          :: i, iter
  character(len=128) :: filename
  
  do i = 1,w
     data(i) = pi/i
  end do

  open (unit=7,file="floattest/initialcond-f90.dat",action="write")
  write (7,*), data(:)
  close (7)
  
  do iter=1, iters
     do i = 1,w
        result(i) = sin((data(i) + data(i) + data(i)) + (data(i) + data(i) + data(i)))
     end do

     write (filename,'(A16,I1,A8)'), "floattest/f.iter", iter-1, "-f90.dat"
     
     open (unit=7,file=filename,action="write")
     write (7,*), result(:)
     close (7)

     do i = 1,w
        data(i) = result(i)
     end do
  end do
  
end program main
