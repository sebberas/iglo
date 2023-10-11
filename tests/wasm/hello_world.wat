;; Hello World
(; Hello World (; Hello World ;) ;)
(module
 (table 0 anyfunc)
 (memory $0 1)
 (data (i32.const 16) "Hello, World\00")
 (export "memory" (memory $0))
 (export "main" (func $main))
 (export "hello_world" (func $hello_world))
 (func $main (; 0 ;) (result i32)
  (local $0 i32)
  (local $1 i32)
  (i32.store offset=4
   (i32.const 0)
   (tee_local $1
    (i32.sub
     (i32.load offset=4
      (i32.const 0)
     )
     (i32.const 16)
    )
   )
  )
  (i32.store offset=12
   (get_local $1)
   (i32.const 0)
  )
  (set_local $0
   (call $hello_world)
  )
  (i32.store offset=4
   (i32.const 0)
   (i32.add
    (get_local $1)
    (i32.const 16)
   )
  )
  (get_local $0)
 )
 (func $hello_world (; 1 ;) (result i32)
  (i32.const 16)
 )
)
