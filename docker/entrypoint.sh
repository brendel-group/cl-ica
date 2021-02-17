#!/bin/bash

# start ssh
service ssh restart

# Create user account
if [ -n "$USER" ]; then
   if [ -z "$USER_HOME" ]; then
      export USER_HOME=/home/$USER
   fi

   if [ -z "$USER_ID" ]; then
      export USER_ID=99
   fi

   if [ -n "$USER_ENCRYPTED_PASSWORD" ]; then
      useradd -M -d $USER_HOME -p $USER_ENCRYPTED_PASSWORD -u $USER_ID $USER > /dev/null
   else
      useradd -M -d $USER_HOME -u $USER_ID $USER > /dev/null
   fi

   # expects a comma-separated string of the form GROUP1:GROUP1ID,GROUP2,GROUP3:GROUP3ID,...
   # (the GROUPID is optional, but needs to be separated from the group name by a ':')
   for i in $(echo $USER_GROUPS | sed "s/,/ /g")
   do
      if [[ $i == *":"* ]]
      then
         addgroup ${i%:*} # > /dev/null
         groupmod -g ${i#*:} ${i%:*} #> /dev/null
         adduser $USER ${i%:*} #> /dev/null
      else
         addgroup $i > /dev/null
         adduser $USER $i > /dev/null
      fi
   done

   # add user to sudo group
   adduser $USER sudo

   # set correct primary group
   if [ -n "$USER_GROUPS" ]; then
      group="$( cut -d ',' -f 1 <<< "$USER_GROUPS" )"
      if [[ $group == *":"* ]]
      then
         usermod -g ${group%:*} $USER &
      else
         usermod -g $group $USER &
      fi
   fi

   # set shell
   if [ -z "$USER_SHELL" ]
   then
       usermod -s "/bin/bash" $USER
   else
       usermod -s $USER_SHELL $USER
   fi

  if [ -n $CWD ]; then cd $CWD; fi
  echo "Running as user $USER"

  # set environment such that gpus can be used even in ssh connections
  echo "export CUDNN_VERSION=$CUDNN_VERSION" >> /etc/profile
  echo "export NVIDIA_REQUIRE_CUDA='$NVIDIA_REQUIRE_CUDA'" >> /etc/profile
  echo "export LIBRARY_PATH=$LIBRARY_PATH" >> /etc/profile
  echo "export LD_PRELOAD=$LD_PRELOAD" >> /etc/profile
  echo "export NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES" >> /etc/profile
  echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /etc/profile
  echo "export NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES" >> /etc/profile
  echo "export PATH=$PATH" >> /etc/profile
  echo "export CUDA_PKG_VERSION=$CUDA_PKG_VERSION" >> /etc/profile

  exec gosu $USER "$@"
else
  if [ -n $CWD ]; then cd $CWD; fi
  echo "Running as default container user"
  exec "$@"
fi
